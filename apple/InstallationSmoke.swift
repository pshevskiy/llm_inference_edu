import CryptoKit
import Foundation

private actor HeldAsset {
    var entered = false
    private var waiter: CheckedContinuation<Void, Never>?
    func fetch(_ source: URL, _ destination: URL, _ asset: ModelAsset) async throws {
        entered = true
        try Data("partial".utf8).write(to: destination)
        await withCheckedContinuation { waiter = $0 }
        try Task.checkCancellation()
    }
    func release() { waiter?.resume(); waiter = nil }
}
@main
struct InstallationSmoke {
    static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func main() async throws {
        let decoder = JSONDecoder()
        let source = try decoder.decode(SourceInventory.self, from: Data(contentsOf: URL(fileURLWithPath: "fixtures/qwen-inventory.json")))
        let artifact = try decoder.decode(CPUArtifact.self, from: Data(contentsOf: URL(fileURLWithPath: "fixtures/cpu-native-artifact.json")))
        let actual = CPURecipe(source: source, artifact: artifact)
        let executable = URL(fileURLWithPath: "target/native/InferenceWorkbench.app/Contents/MacOS/InferenceWorkbench")
        let library = URL(fileURLWithPath: "target/release/libllm_cpu_bridge.a")
        if CommandLine.arguments.contains("--network-small") {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("llm-transfer-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            defer { try? FileManager.default.removeItem(at: directory) }
            let output = directory.appendingPathComponent("LICENSE")
            let url = URL(string: "https://huggingface.co/\(source.repository)/resolve/\(source.revision)/LICENSE")!
            try await ModelInstaller.download(url, output, source.assets["LICENSE"]!)
            let matches = try ModelInstaller.hash(output, asset: source.assets["LICENSE"]!)
            precondition(matches)
            // Deliberately tiny admission exercises observed-overrun cancellation,
            // with a bounded real public payload rather than downloading weights.
            do {
                try await ModelInstaller.download(url, directory.appendingPathComponent("too-large"),
                    ModelAsset(bytes: 1, sha256: source.assets["LICENSE"]!.sha256))
                preconditionFailure("download overrun admitted")
            } catch { precondition(error as? InstallFailure == .integrity("too-large")) }
            let cancelled = Task { try await ModelInstaller.download(url, directory.appendingPathComponent("cancelled"), source.assets["LICENSE"]!) }
            cancelled.cancel()
            do { try await cancelled.value; preconditionFailure("transfer cancellation ignored") }
            catch { precondition(error as? InstallFailure == .cancelled) }
            print("Public LICENSE-only transfer/size/hash/overrun/pre-admission cancellation passed")
            return
        }
        if CommandLine.arguments.contains("--local") {
            let local = URL(fileURLWithPath: "models/qwen2.5-0.5b-instruct/\(source.revision)", isDirectory: true)
            let installer = ModelInstaller()
            let installed = try await installer.install(actual, root: URL(fileURLWithPath: "models/installed-qwen", isDirectory: true),
                fetch: { url, destination, _ in
                    try Task.checkCancellation()
                    try FileManager.default.copyItem(at: local.appendingPathComponent(url.lastPathComponent), to: destination)
                })
            let verified = try await Task.detached { try ModelInstaller.verify(actual, directory: installed.directory) }.value
            let execution = try await Task.detached { try ModelInstaller.verifyExecution(artifact, library: library, executable: executable) }.value
            let discovered = try decoder.decode(HubModel.self, from: Data("{\"id\":\"\(source.repository)\",\"sha\":\"\(source.revision)\"}".utf8))
            precondition(actual.compatibility(discovered, installation: verified, execution: execution).status == .runnable)
            print("Pinned local copy verified and atomically installed at \(installed.directory.path)")
            return
        }
        // Same trusted identity but tiny independent payloads for offline tests;
        // these test fixtures are never numerically loaded or claimed as Qwen.
        var payloads: [String: Data] = [:]
        var assets: [String: ModelAsset] = [:]
        for name in source.assets.keys {
            let data = Data("test-\(name)".utf8); payloads[name] = data
            assets[name] = ModelAsset(bytes: UInt64(data.count), sha256: digest(data))
        }
        let tiny = SourceInventory(repository: source.repository, revision: source.revision, license: source.license,
            format: source.format, shards: source.shards, assets: assets)
        // Test-only execution bytes/digests: test hash admission, not inference.
        let testLibrary = Data("offline library fixture".utf8)
        let testExecutable = Data("offline executable fixture".utf8)
        let testingArtifact = CPUArtifact(id: artifact.id, librarySHA256: digest(testLibrary), qwenQualified: true,
            executableSHA256: digest(testExecutable), validation: "offline fixtures only; not a qualified model")
        let recipe = CPURecipe(source: tiny, artifact: testingArtifact)
        let data = payloads
        let fetch: ModelInstaller.Fetch = { url, destination, _ in try data[url.lastPathComponent]!.write(to: destination) }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("llm-install-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let mockLibrary = root.appendingPathComponent("fixture.a")
        let mockExecutable = root.appendingPathComponent("fixture-executable")
        try testLibrary.write(to: mockLibrary); try testExecutable.write(to: mockExecutable)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: mockExecutable.path)
        let execution = try ModelInstaller.verifyExecution(testingArtifact, library: mockLibrary, executable: mockExecutable)
        let installer = ModelInstaller()
        do { _ = try await installer.install(recipe, root: root, availableBytes: 0, fetch: fetch); preconditionFailure("space check") }
        catch { precondition(error as? InstallFailure == .insufficientSpace) }
        let installed = try await installer.install(recipe, root: root, availableBytes: 1_000_000_000, fetch: fetch)
        let fetchedAgain = try await installer.install(recipe, root: root, fetch: { _,_,_ in preconditionFailure("existing install redownload") })
        precondition(installed.directory == fetchedAgain.directory)
        let exact = try decoder.decode(HubModel.self, from: Data("{\"id\":\"\(source.repository)\",\"sha\":\"\(source.revision)\"}".utf8))
        precondition(recipe.compatibility(exact).status == .recipeAvailable)
        precondition(recipe.compatibility(exact, installation: installed).status == .recipeAvailable)
        let badLibrary = root.appendingPathComponent("replaced-library.a")
        try Data("not the qualified library".utf8).write(to: badLibrary)
        do { _ = try ModelInstaller.verifyExecution(testingArtifact, library: badLibrary, executable: mockExecutable); preconditionFailure("unverified execution admitted") }
        catch { precondition(error as? InstallFailure == .integrity("execution artifact")) }
        precondition(recipe.compatibility(exact, installation: installed, execution: execution).status == .runnable)
        precondition(actual.compatibility(exact, installation: installed).status == .recipeAvailable) // different payload manifest
        let pendingArtifact = CPUArtifact(id: testingArtifact.id, librarySHA256: testingArtifact.librarySHA256,
            qwenQualified: false, executableSHA256: testingArtifact.executableSHA256, validation: "Qwen pending")
        let pendingRecipe = CPURecipe(source: tiny, artifact: pendingArtifact)
        let pendingSource = try ModelInstaller.verify(pendingRecipe, directory: installed.directory)
        let pendingExecution = try ModelInstaller.verifyExecution(pendingArtifact, library: mockLibrary, executable: mockExecutable)
        precondition(pendingRecipe.compatibility(exact, installation: pendingSource, execution: pendingExecution).status == .recipeAvailable)
        let mlx = try decoder.decode(HubModel.self, from: Data(#"{"id":"other/Qwen-4bit","library_name":"mlx"}"#.utf8))
        let c = actual.compatibility(mlx, installation: installed)
        precondition(c.status == .discoveryOnly && c.source == nil && c.execution == nil)
        let different = try decoder.decode(HubModel.self, from: Data("{\"id\":\"\(source.repository)\",\"sha\":\"different\"}".utf8))
        precondition(actual.compatibility(different).status == .unsupported)
        let unknown = try decoder.decode(HubModel.self, from: Data("{\"id\":\"\(source.repository)\"}".utf8))
        precondition(actual.compatibility(unknown).status == .discoveryOnly)
        try Data("corrupt".utf8).write(to: installed.directory.appendingPathComponent("LICENSE"))
        do { _ = try ModelInstaller.verify(recipe, directory: installed.directory); preconditionFailure("corruption admitted") }
        catch { precondition(error as? InstallFailure == .integrity("LICENSE")) }
        let damaged = try Data(contentsOf: installed.directory.appendingPathComponent("LICENSE"))
        do { _ = try await installer.install(recipe, root: root, fetch: fetch); preconditionFailure("existing corrupt overwritten") }
        catch { precondition(error as? InstallFailure == .integrity("LICENSE")) }
        let unchanged = try Data(contentsOf: installed.directory.appendingPathComponent("LICENSE"))
        precondition(unchanged == damaged)

        let interruptedRoot = root.appendingPathComponent("interrupted")
        do {
            _ = try await installer.install(recipe, root: interruptedRoot, availableBytes: 1_000_000_000,
                fetch: { _, destination, _ in try Data("wrong".utf8).write(to: destination) })
            preconditionFailure("bad hash accepted")
        } catch { precondition(error is InstallFailure) }
        let emptyAfterFailure = try FileManager.default.contentsOfDirectory(atPath: interruptedRoot.path)
        precondition(emptyAfterFailure.isEmpty)
        let held = HeldAsset()
        let waiting = Task {
            try await installer.install(recipe, root: interruptedRoot, availableBytes: 1_000_000_000,
                fetch: { try await held.fetch($0, $1, $2) })
        }
        while !(await held.entered) { await Task.yield() }
        do { _ = try await installer.install(recipe, root: interruptedRoot, fetch: fetch); preconditionFailure("double install") }
        catch { precondition(error as? InstallFailure == .busy) }
        await installer.cancel(); await held.release()
        do { _ = try await waiting.value; preconditionFailure("cancel admitted") }
        catch { precondition(error as? InstallFailure == .cancelled) }
        let emptyAfterCancel = try FileManager.default.contentsOfDirectory(atPath: interruptedRoot.path)
        precondition(emptyAfterCancel.isEmpty)
        _ = try await installer.install(recipe, root: interruptedRoot, availableBytes: 1_000_000_000, fetch: fetch)
        precondition(!CPURecipe.filenameValid("../weights"))
        precondition(!CPURecipe.filenameValid("/tmp/weights"))
        print("Offline identity/pinning/disk/hash/atomic/cancel/recovery installation contracts passed")
    }
}
