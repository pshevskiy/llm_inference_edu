// Mac controller integration test using the real Rust bridge. This does not
// execute SwiftUI, iOS lifecycle delivery, or a physical phone's memory budget.
import Foundation

@main
struct PhoneSessionSmoke {
    @MainActor
    static func settle(_ session: PhoneSession) async throws {
        let deadline = Date().addingTimeInterval(5)
        while session.busy && Date() < deadline {
            try await Task.sleep(for: .milliseconds(2))
        }
        precondition(!session.busy, "bounded wait timed out")
    }
    @MainActor
    static func main() async throws {
        precondition(CommandLine.arguments.count == 2)
        let session = PhoneSession(fixture: URL(fileURLWithPath: CommandLine.arguments[1]))
        session.load()
        session.sceneActive(false) // READY must not undo background pause
        try await settle(session)
        precondition(session.ready && session.paused && !session.canPrepare)
        session.sceneActive(true)
        precondition(session.paused)
        session.resume()
        precondition(session.state == "Ready" && session.canPrepare)
        session.prepare(); session.prepare() // second operation is not queued
        try await settle(session)
        precondition(session.processed == 3 && session.history == 3)
        session.step(); session.step()
        try await settle(session)
        precondition(session.tokens == [41] && session.processed == 3 && session.history == 4)
        session.sceneActive(false)
        session.step() // no work admitted in background
        precondition(!session.busy)
        session.sceneActive(true)
        precondition(session.paused && !session.canStep)
        session.resume()
        for (index, token) in [UInt32(53), 41, 50].enumerated() {
            session.step()
            try await settle(session)
            precondition(session.tokens.last == token)
            precondition(session.processed == UInt32(index + 4) && session.history == UInt32(index + 5))
        }
        session.step()
        try await settle(session)
        precondition(session.state == "Finished" && !session.canStep)
        session.pause(); session.resume()
        precondition(session.state == "Finished" && !session.canStep)
        session.pause(); session.reset()
        try await settle(session)
        precondition(session.paused && session.tokens.isEmpty && session.history == 0)
        session.resume(); session.prepare()
        try await settle(session)
        session.step(); session.pause() // cancellation may race the current token boundary
        try await settle(session)
        precondition(session.ready && session.paused)
        precondition(session.tokens.isEmpty || session.tokens == [41])
        precondition(session.processed == 3 && session.history == 3 + UInt32(session.tokens.count))
        session.resume(); session.step()
        try await settle(session)
        precondition(session.tokens == [41] || session.tokens == [41, 53])
        session.step(); session.memoryWarning() // invalidate queued callbacks
        try await Task.sleep(for: .milliseconds(100))
        precondition(session.state == "Unloaded" && session.tokens.isEmpty && session.history == 0)
        session.load()
        try await settle(session)
        precondition(session.ready)
        session.prepare(); session.sceneActive(false)
        try await settle(session)
        precondition(session.ready && session.paused && session.processed <= session.history)
        precondition(session.history == 0 || session.history == 3)
        session.sceneActive(true); session.resume()
        session.reset(); session.sceneActive(false)
        try await settle(session)
        precondition(session.ready && session.paused && session.history == 0)
        session.sceneActive(true)
        precondition(!session.canPrepare)
        session.resume(); session.prepare()
        try await settle(session)
        session.step()
        try await settle(session)
        precondition(session.tokens == [41] && session.processed == 3 && session.history == 4)
        session.failForSmokeTest()
        precondition(session.tokens.isEmpty && session.processed == 0 && session.history == 0)
        precondition(session.state == "Failed" && !session.ready && !session.canPrepare && !session.canStep)
        session.sceneActive(false); session.sceneActive(true); session.pause()
        precondition(session.state == "Failed" && !session.canPause)
        session.unload()
        try await Task.sleep(for: .milliseconds(100))
        session.load(); session.unload() // invalidate a pending fixture read
        try await Task.sleep(for: .milliseconds(100))
        precondition(session.state == "Unloaded" && !session.ready && !session.busy)
        let missing = PhoneSession(fixture: URL(fileURLWithPath: "/missing/fixture.json"))
        missing.load()
        try await settle(missing)
        precondition(missing.state == "Failed" && missing.canLoad)
        print("Mac controller checks passed: tiny tokens [41,53,41,50], exact 3/4→6/7; pause/background/reset/unload/stale callbacks/memory-warning handler/terminal-failure display invalidation. Not iOS execution.")
    }
}
