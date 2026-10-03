import AppKit
import Foundation
import LLMCPUBridge
import SwiftUI

@MainActor
final class Workbench: ObservableObject {
    @Published var query = "Qwen"
    @Published var publisher = ""
    @Published var library = "mlx"
    @Published var results: [HubModel] = []
    @Published var searching = false
    @Published var hasNext = false
    @Published var installing = false
    @Published var installed: VerifiedInstallation?
    @Published var execution: VerifiedExecution?
    @Published var state = "Unloaded"
    @Published var error = ""
    @Published var prompt = "Hello!"
    @Published var submitted = ""
    @Published var answer = ""
    @Published var processed: UInt32 = 0
    @Published var history: UInt32 = 0
    @Published var generating = false
    @Published var ready = false
    @Published var tiny = false
    @Published var outputLimit = 4
    @Published var waiting = false
    @Published var sent = false
    private let searcher: HubSearch
    private let installer = ModelInstaller()
    private var searchTask: Task<Void, Never>?
    private var installTask: Task<Void, Never>?
    private var epoch: UInt64 = 0
    private var searchEpoch: UInt64 = 0
    private var client: CPUClient?
    let recipe: CPURecipe
    let resources: URL
    let installRoot: URL

    init(searcher: HubSearch = HubSearch(), verifyOnLaunch: Bool = true) {
        self.searcher = searcher
        resources = Bundle.main.resourceURL!
        let decode = JSONDecoder()
        do {
            let source = try decode.decode(SourceInventory.self, from: Data(contentsOf: resources.appendingPathComponent("qwen-inventory.json")))
            let artifact = try decode.decode(CPUArtifact.self, from: Data(contentsOf: resources.appendingPathComponent("cpu-native-artifact.json")))
            recipe = CPURecipe(source: source, artifact: artifact)
        } catch { fatalError("Invalid bundled recipe: \(error)") }
        // The user-selected build workspace is recorded by the build tool; no
        // prompt/transcript is persisted. App-created model files stay ignored.
        guard let root = try? String(contentsOf: resources.appendingPathComponent("workspace.txt"), encoding: .utf8), root.hasPrefix("/") else {
            fatalError("Missing build workspace")
        }
        installRoot = URL(fileURLWithPath: root.trimmingCharacters(in: .whitespacesAndNewlines))
            .appendingPathComponent("models/installed-qwen", isDirectory: true)
        if verifyOnLaunch { Task {
            let recipe = self.recipe, directory = installRoot.appendingPathComponent(recipe.source.revision)
            let library = resources.appendingPathComponent("libllm_cpu_bridge.a")
            let executable = Bundle.main.executableURL!
            do {
                execution = try await Task.detached { try ModelInstaller.verifyExecution(recipe.artifact, library: library, executable: executable) }.value
                if FileManager.default.fileExists(atPath: directory.path) {
                    installed = try await Task.detached { try ModelInstaller.verify(recipe, directory: directory) }.value
                }
            } catch { self.error = "Verification failed: \(error)" }
        } }
    }
    func compatibility(_ model: HubModel) -> ModelCompatibility {
        recipe.compatibility(model, installation: installed, execution: execution)
    }
    func search(next: Bool = false) {
        searchEpoch &+= 1
        let id = searchEpoch
        searchTask?.cancel()
        searching = true; error = ""
        let options = HubQuery(text: query, publisher: publisher.isEmpty ? nil : publisher,
            library: library.isEmpty ? nil : library)
        if !next { results = []; hasNext = false }
        searchTask = Task {
            do {
                let page = try await (next ? searcher.next() : searcher.search(options))
                guard id == searchEpoch else { return }
                results = page.models; hasNext = page.next != nil
            } catch {
                guard id == searchEpoch else { return }
                if !Task.isCancelled { self.error = "Search: \(error)" }
            }
            guard id == searchEpoch else { return }
            searching = false
        }
    }
    func cancelSearch() {
        searchEpoch &+= 1; searchTask?.cancel(); searching = false
    }
    func install() {
        guard !installing else { return }
        installing = true; error = ""
        let recipe = self.recipe, root = installRoot
        installTask = Task {
            do { installed = try await installer.install(recipe, root: root) }
            catch { self.error = "Installation: \(error)" }
            installing = false
        }
    }
    func cancelInstall() { installTask?.cancel() }
    func load(demo: Bool) {
        guard client == nil, !waiting else { return }
        guard demo || (installed != nil && execution != nil && recipe.artifact.qwenQualified) else {
            error = "The current source and execution build need verification before loading."; return
        }
        error = ""; waiting = true; state = "Loading"; tiny = demo
        prompt = demo ? "1,5,7" : "Hello!"
        let identity = epoch
        let payload: Data
        do { payload = demo ? try Data(contentsOf: resources.appendingPathComponent("tiny-model.json")) : Data(installed!.directory.path.utf8) }
        catch { self.error = "Model source: \(error)"; waiting = false; state = "Failed"; return }
        do {
            let newClient = try CPUClient(kind: demo ? 0 : 1, payload: payload)
            client = newClient
            precondition(newClient.waitReady { [weak self] result in
                // CPUClient promises synchronous delivery on DispatchQueue.main.
                MainActor.assumeIsolated {
                    guard let self, identity == self.epoch else { return }
                    self.waiting = false
                    switch result {
                    case .success: self.ready = true; self.state = "Ready"
                    case let .failure(e): self.error = "Load failed (\(e.code))"; self.state = "Failed"
                    }
                }
            })
        } catch { self.error = "Load admission: \(error)"; waiting = false; state = "Failed" }
    }
    func unload() {
        epoch &+= 1; client?.close(); client = nil
        ready = false; waiting = false; generating = false; state = "Unloaded"
        processed = 0; history = 0; sent = false; submitted = ""; answer = ""
    }
    func send() {
        guard ready && !waiting && !sent && state != "Paused", let client else { return }
        error = ""; waiting = true; state = "Processing prompt"
        let identity = epoch
        let completion: @Sendable (Result<CPUEvent, CPUFailure>) -> Void = { [weak self] result in
            MainActor.assumeIsolated {
                guard let self, identity == self.epoch else { return }
                self.waiting = false
                switch result {
                case let .success(event):
                    self.processed = event.processed; self.history = event.history
                    if event.kind == UInt32(AIE_CANCEL_EVENT) {
                        self.sent = event.history > 0; self.state = "Paused"; self.generating = false
                    } else { self.sent = true; self.generating = true; self.state = "Generating"; self.step() }
                case let .failure(e):
                    if e.code == UInt32(AIE_INVALID_INPUT) {
                        self.error = "Prompt exceeds the validated budget or contains invalid input."; self.state = "Ready"
                    } else {
                        self.error = "Engine failed (\(e.code)); unload to recover."; self.state = "Failed"; self.ready = false; self.processed = 0
                    }
                }
            }
        }
        submitted = prompt; answer = ""
        let admitted: Bool
        if tiny {
            guard prompt.utf8.prefix(257).count <= 256 else { error = "Token-ID prompt is too long."; waiting = false; return }
            let fields = prompt.split(separator: ",", omittingEmptySubsequences: false)
            let ids = fields.compactMap { UInt32($0.trimmingCharacters(in: .whitespaces)) }
            guard !fields.isEmpty && ids.count == fields.count else { error = "Enter comma-separated token IDs."; waiting = false; return }
            admitted = client.begin(ids: ids, maxNew: UInt32(outputLimit), completion: completion)
        } else { admitted = client.begin(text: prompt, maxNew: UInt32(outputLimit), completion: completion) }
        if !admitted { waiting = false; error = "Engine is busy; try again after the current response." }
    }
    private func step() {
        guard generating, !waiting, let client else { return }
        waiting = true
        let identity = epoch
        guard client.next({ [weak self] result in
            MainActor.assumeIsolated {
                guard let self, identity == self.epoch else { return }
                self.waiting = false
                switch result {
                case let .failure(e): self.error = "Generation failed (\(e.code)); unload to recover."; self.generating = false; self.ready = false; self.processed = 0; self.state = "Failed"
                case let .success(event):
                    self.processed = event.processed; self.history = event.history
                    self.answer += event.text
                    switch event.kind {
                    case UInt32(AIE_TOKEN):
                        if self.generating { self.step() }
                    case UInt32(AIE_CANCEL_EVENT): self.generating = false; self.state = "Paused"
                    case UInt32(AIE_STOPPED): self.generating = false; self.state = "Finished"
                    default: self.generating = false; self.state = "Failed"; self.error = "Unexpected engine event."
                    }
                }
            }
        }) else { waiting = false; generating = false; error = "Engine request was not admitted."; return }
    }
    func pause() {
        generating = false; state = "Paused"
        if let client { _ = client.setCancelled(true) }
    }
    func resume() {
        guard !waiting, state == "Paused", let client else { return }
        _ = client.setCancelled(false)
        if !sent { state = "Ready"; submitted = ""; answer = ""; return }
        generating = true; state = "Generating"; step()
    }
    func newChat() {
        guard ready, !waiting, !generating, let client else { return }
        waiting = true; let identity = epoch
        precondition(client.reset { [weak self] result in
            MainActor.assumeIsolated {
                guard let self, identity == self.epoch else { return }
                self.waiting = false
                switch result {
                case .success: self.sent = false; self.submitted = ""; self.answer = ""; self.processed = 0; self.history = 0; self.state = "Ready"
                case let .failure(e): self.error = "Reset failed (\(e.code)); unload to recover."; self.ready = false; self.state = "Failed"; self.processed = 0
                }
            }
        })
    }
}

struct WorkbenchView: View {
    @ObservedObject var model: Workbench
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Inference Workbench").font(.title.bold())
                Spacer()
                Text(model.state).foregroundStyle(.secondary)
                Button("Unload") { model.unload() }.disabled(model.state == "Unloaded")
            }
            if !model.error.isEmpty { Text(model.error).foregroundStyle(.red).textSelection(.enabled) }
            TabView {
                search.tabItem { Label("Search", systemImage: "magnifyingglass") }
                models.tabItem { Label("Models", systemImage: "internaldrive") }
                chat.tabItem { Label("Chat", systemImage: "bubble.left") }
                diagnostics.tabItem { Label("Diagnostics", systemImage: "info.circle") }
            }
        }.padding(20).frame(minWidth: 780, minHeight: 540)
    }
    private var search: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Find public models. Discovery is separate from local execution.").foregroundStyle(.secondary)
            HStack {
                TextField("Model name", text: $model.query)
                TextField("Publisher (optional)", text: $model.publisher)
                TextField("Library (blank for all)", text: $model.library)
                Button("Search") { model.search() }
                if model.searching { Button("Cancel") { model.cancelSearch() }; ProgressView().controlSize(.small) }
            }
            List(model.results) { item in
                let result = model.compatibility(item)
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.id).font(.headline)
                    Text("\(item.library_name ?? "Unknown library") · \(result.status.rawValue)").foregroundStyle(.secondary)
                    Text(result.explanation).font(.caption)
                }.padding(.vertical, 4)
            }
            Button("Next page") { model.search(next: true) }.disabled(!model.hasNext || model.searching)
        }.padding()
    }
    private var models: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(model.recipe.source.repository).font(.headline)
            Text("Source revision: \(model.recipe.source.revision)").textSelection(.enabled)
            Text("Execution: Rust FP32 CPU · short prompts · Neural Engine unqualified")
            Text(model.recipe.artifact.validation).font(.caption).foregroundStyle(.secondary)
            Text(model.installed == nil ? "Source: awaiting installation/verification" : "Source: pinned files verified")
            Text(model.execution == nil ? "Execution: awaiting verification" : "Execution: current binary and archive verified")
            if let installed = model.installed { Text(installed.directory.path).font(.caption).textSelection(.enabled) }
            HStack {
                Button("Install pinned source (~1 GB)") { model.install() }.disabled(model.installing || model.installed != nil)
                if model.installing { ProgressView().controlSize(.small); Button("Cancel install") { model.cancelInstall() } }
                Button("Load Qwen") { model.load(demo: false) }.disabled(model.installed == nil || model.execution == nil || !model.recipe.artifact.qwenQualified || model.state != "Unloaded")
            }
            Divider()
            Text("Synthetic fixture").font(.headline)
            Text("Small offline token-ID demo for inspecting generation state.").foregroundStyle(.secondary)
            Button("Load tiny demo") { model.load(demo: true) }.disabled(model.state != "Unloaded")
            Spacer()
        }.padding()
    }
    private var chat: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(model.tiny ? "Synthetic token-ID demo" : "Plain prompt · CPU only · one short turn per chat").font(.headline)
            Text(model.tiny ? "Budget: 32 total positions; enter comma-separated IDs." : "Validated budget: 8 total positions, including prompt and generated tokens. Longer prompts are rejected.").foregroundStyle(.secondary)
            HStack {
                Stepper("Output limit: \(model.outputLimit)", value: $model.outputLimit, in: 1...4).frame(width: 230)
                Text("Evaluated \(model.processed) · history \(model.history)").monospacedDigit().foregroundStyle(.secondary)
            }
            if !model.submitted.isEmpty { Text("You: \(model.submitted)").textSelection(.enabled) }
            ScrollView { Text(model.answer.isEmpty ? "The answer appears here." : model.answer).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled) }
            TextField(model.tiny ? "1,5,7" : "Short prompt", text: $model.prompt).textFieldStyle(.roundedBorder).disabled(model.sent)
            HStack {
                Button("Send") { model.send() }.disabled(!model.ready || model.waiting || model.sent || model.state == "Paused")
                Button("Pause") { model.pause() }.disabled(!model.generating && model.state != "Processing prompt")
                Button("Resume") { model.resume() }.disabled(model.state != "Paused" || model.waiting)
                Button("New chat") { model.newChat() }.disabled(!model.ready || model.waiting || model.generating)
                if model.waiting { ProgressView().controlSize(.small) }
            }
        }.padding()
    }
    private var diagnostics: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Execution and validation").font(.headline)
            Text("Own Rust CPU forward pass · greedy sampling · one model worker")
            Text("Allocated Qwen cache: 32 positions; admitted app requests: 8 positions")
            Text("Mac native build. Physical iPhone and iPad validation pending.")
            Text("Neural Engine acceleration and energy measurements unqualified.")
            Text("Source files and execution bytes are verified before loading. Chat runs offline after installation.")
            Text("Archive SHA256: \(model.recipe.artifact.librarySHA256)").font(.caption).textSelection(.enabled)
            Text("Executable SHA256: \(model.recipe.artifact.executableSHA256)").font(.caption).textSelection(.enabled)
            Text("Cancellation pauses at token boundaries. Unload releases resources asynchronously; retry if the previous worker is still closing.").foregroundStyle(.secondary)
            Spacer()
        }.padding()
    }
}

@main
struct WorkbenchApp: App {
    @StateObject private var model: Workbench
    init() {
        let args = CommandLine.arguments
        if args.count == 2 && args[1] == "--state-smoke" { WorkbenchStateSmoke.run() }
        if args.count >= 2 && args[1] == "--smoke" {
            do { try BoundarySmoke.run([args[0]] + Array(args.dropFirst(2))) }
            catch { print("Native smoke failed: \(error)"); exit(1) }
        }
        _model = StateObject(wrappedValue: Workbench())
    }
    var body: some Scene {
        WindowGroup {
            WorkbenchView(model: model).onDisappear {
                model.unload(); model.cancelSearch(); model.cancelInstall()
            }
        }
            .commands { CommandGroup(replacing: .newItem) { Button("New chat") { model.newChat() } } }
    }
}
