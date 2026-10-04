import Combine
import Foundation
import LLMCPUBridge

// Concrete tiny-demo controller. Rust owns every model/cache/token; the UI holds
// at most four displayed tokens. Pausing and operation admission are independent.
@MainActor
final class PhoneSession: ObservableObject {
    @Published private(set) var state = "Unloaded"
    @Published private(set) var busy = false
    @Published private(set) var paused = false
    @Published private(set) var ready = false
    @Published private(set) var tokens: [UInt32] = []
    @Published private(set) var processed: UInt32 = 0
    @Published private(set) var history: UInt32 = 0
    @Published private(set) var message = "Load the bundled engine demo to begin."
    private var client: CPUClient?
    private var epoch: UInt64 = 0
    @Published private var active = true
    private var finished = false
    private let fixture: URL
    var canLoad: Bool { active && !busy && client == nil }
    var canPrepare: Bool { active && ready && !busy && !paused && history == 0 }
    var canStep: Bool { active && ready && !busy && !paused && history > 0 && !finished }
    var canReset: Bool { active && ready && !busy }
    var hasSession: Bool { client != nil || busy }
    var canPause: Bool { hasSession && !paused && state != "Failed" }

    init(fixture: URL) { self.fixture = fixture }

    func load() {
        guard canLoad else { return }
        epoch &+= 1
        let identity = epoch, source = fixture
        busy = true; state = "Loading"; message = "Loading the small demo off the main thread."
        Task { [self] in
            do {
                let data = try await Task.detached {
                    let file = try FileHandle(forReadingFrom: source)
                    defer { try? file.close() }
                    let bytes = try file.read(upToCount: 8 * 1024 * 1024 + 1) ?? Data()
                    guard !bytes.isEmpty && bytes.count <= 8 * 1024 * 1024 else {
                        throw CocoaError(.fileReadTooLarge)
                    }
                    return bytes
                }.value
                guard identity == self.epoch else { return }
                let engine = try CPUClient(kind: 0, payload: data)
                self.client = engine
                if self.paused || !self.active { _ = engine.setCancelled(true) }
                guard engine.waitReady({ [weak self] result in
                    MainActor.assumeIsolated {
                        guard let self, identity == self.epoch else { return }
                        self.busy = false
                        switch result {
                        case let .success(event):
                            guard event.kind == UInt32(AIE_READY) else { self.fail("Unexpected load event."); return }
                            self.ready = true
                            self.state = self.paused ? "Paused" : "Ready"
                            self.message = self.paused ? "Resume explicitly when ready." : "Prepare the three-token prompt."
                        case let .failure(error): self.fail("Load failed (\(error.code)). Unload, then retry.")
                        }
                    }
                }) else { self.fail("Load response was not admitted."); return }
            } catch {
                guard identity == self.epoch else { return }
                self.fail("Load unavailable: \(error). If a prior session is closing, retry after it releases.")
            }
        }
    }
    func prepare() {
        guard canPrepare, let client else { return }
        submit { client.begin(ids: [1, 5, 7], maxNew: 4, completion: $0) }
    }
    func step() {
        guard canStep, let client else { return }
        submit { client.next($0) }
    }
    private func submit(_ operation: (@escaping @Sendable (Result<CPUEvent, CPUFailure>) -> Void) -> Bool) {
        let identity = epoch
        busy = true
        guard operation({ [weak self] result in
            MainActor.assumeIsolated {
                guard let self, identity == self.epoch else { return }
                self.busy = false
                switch result {
                case let .failure(error): self.fail("Engine failed (\(error.code)); unload to recover.")
                case let .success(event):
                    self.processed = event.processed; self.history = event.history
                    switch event.kind {
                    case UInt32(AIE_STARTED):
                        self.state = self.paused ? "Paused" : "Prepared"
                        self.message = "The prompt is evaluated. Request one token at a time."
                    case UInt32(AIE_TOKEN):
                        guard let token = event.token, self.tokens.count < 4 else { self.fail("Invalid token event."); return }
                        self.tokens.append(token)
                        self.state = self.paused ? "Paused" : "Prepared"
                        self.message = "The emitted token remains pending until the next step."
                    case UInt32(AIE_STOPPED): self.finished = true; self.state = self.paused ? "Paused" : "Finished"; self.message = "Output limit reached. Reset to run again."
                    case UInt32(AIE_CANCEL_EVENT): self.paused = true; self.state = "Paused"; self.message = "Exact token history is retained."
                    case UInt32(AIE_RESET_EVENT):
                        // The worker's reset clears cancellation. Restore it if
                        // the UI paused or backgrounded while reset was in flight.
                        if self.paused || !self.active {
                            guard let client = self.client, client.setCancelled(true) == UInt32(AIE_OK) else {
                                self.fail("Pause after reset unavailable."); return
                            }
                        }
                        self.finished = false; self.tokens = []; self.processed = 0; self.history = 0
                        self.state = self.paused ? "Paused" : "Ready"; self.message = "Session reset."
                    default: self.fail("Unexpected engine event.")
                    }
                }
            }
        }) else { busy = false; message = "Engine is busy; no second request was queued."; return }
    }
    func pause() {
        guard canPause else { return }
        paused = true
        if let client, client.setCancelled(true) != UInt32(AIE_OK) { fail("Cancellation unavailable; unload to recover."); return }
        state = "Paused"; message = "Paused. Work already in flight may finish its token boundary."
    }
    func resume() {
        guard active && ready && !busy && paused, let client else { return }
        guard client.setCancelled(false) == UInt32(AIE_OK) else { fail("Resume unavailable."); return }
        paused = false
        state = finished ? "Finished" : (history == 0 ? "Ready" : "Prepared")
        message = "Resumed. Request the next step explicitly."
    }
    func reset() {
        guard canReset, let client else { return }
        submit { client.reset($0) }
    }
    func sceneActive(_ enabled: Bool) {
        active = enabled
        if !enabled { pause() }
    }
    func unload(reason: String = "Session closed. Model release is asynchronous.") {
        epoch &+= 1; client?.close(); client = nil
        busy = false; ready = false; paused = false; finished = false; tokens = []; processed = 0; history = 0
        state = "Unloaded"; message = reason
    }
    func memoryWarning() { unload(reason: "Memory warning: the session was closed. Load again when appropriate.") }
    private func fail(_ text: String) {
        busy = false; ready = false; finished = false
        processed = 0; history = 0; tokens = []
        state = "Failed"; message = text + " Cache state unavailable."
    }
    #if PHONE_SESSION_SMOKE
    // Controller state regression only; excluded from shipping app builds.
    func failForSmokeTest() {
        fail("Synthetic terminal engine failure. Unload, then reload.")
    }
    #endif
}
