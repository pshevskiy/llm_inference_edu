import Foundation
import Dispatch

private actor SearchReplacement {
    var entered = false
    var finishedOld = false
    private var waiter: CheckedContinuation<Void, Never>?
    func fetch(_ url: URL) async -> HubResponse {
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems!
        if items.contains(where: { $0.name == "search" && $0.value == "old" }) {
            entered = true
            await withCheckedContinuation { waiter = $0 }
            finishedOld = true
            return HubResponse(status: 200, body: Data(#"[{"id":"old/model"}]"#.utf8))
        }
        return HubResponse(status: 200, body: Data(#"[{"id":"new/model"}]"#.utf8),
            link: #"<https://huggingface.co/api/models?cursor=new>; rel="next""#)
    }
    func release() { waiter?.resume(); waiter = nil }
}
@MainActor
struct WorkbenchStateSmoke {
    static func until(_ test: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(5)
        while !test() {
            precondition(ContinuousClock.now < deadline, "controller smoke timeout")
            try await Task.sleep(for: .milliseconds(2))
        }
    }
    static func run() -> Never {
        Task { @MainActor in
            do {
                let controlled = SearchReplacement()
                let searcher = HubSearch(transport: { await controlled.fetch($0) })
                let model = Workbench(searcher: searcher, verifyOnLaunch: false)
                model.query = "old"; model.search()
                while !(await controlled.entered) { await Task.yield() }
                model.cancelSearch(); model.query = "new"; model.search()
                try await until { !model.searching }
                precondition(model.results.map(\.id) == ["new/model"] && model.hasNext)
                await controlled.release()
                while !(await controlled.finishedOld) { await Task.yield() }
                await Task.yield()
                precondition(model.results.map(\.id) == ["new/model"] && model.hasNext && model.error.isEmpty)
                model.load(demo: true)
                try await until { model.ready }
                // Empty-history pause is the pre-admission cancellation state.
                // Resume must clear the Rust flag and restore Ready without a
                // phantom history token, before a valid prompt can be admitted.
                model.pause(); precondition(model.state == "Paused" && !model.sent && model.history == 0)
                model.send(); precondition(model.submitted.isEmpty)
                model.resume(); precondition(model.state == "Ready" && !model.sent)
                model.send()
                try await until { model.state == "Finished" }
                precondition(model.answer == "41 53 41 50 " && model.processed == 6 && model.history == 7)
                model.newChat(); try await until { !model.waiting }
                precondition(model.state == "Ready" && model.history == 0 && model.answer.isEmpty)
                model.prompt = "99"; model.send(); try await until { !model.waiting }
                precondition(!model.sent && model.history == 0 && model.state == "Ready" && !model.error.isEmpty)
                model.unload(); precondition(!model.ready && model.state == "Unloaded" && model.history == 0)
                print("Offline UI controller cancel/replacement/pause-empty/resume/generate/reset/reject/unload passed")
                exit(0)
            } catch { print("Controller smoke failed: \(error)"); exit(1) }
        }
        dispatchMain()
    }
}
