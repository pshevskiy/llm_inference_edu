// Development native-boundary parity/lifecycle check, not an app or UI.
import Dispatch
import Foundation
import LLMCPUBridge

#if !WORKBENCH
@main
#endif
struct BoundarySmoke {
    static func checked(_ result: Result<CPUEvent, CPUFailure>, kind: UInt32) -> CPUEvent {
        dispatchPrecondition(condition: .onQueue(.main))
        let event = try! result.get()
        precondition(event.kind == kind)
        return event
    }
    static func main() throws {
        try run(CommandLine.arguments)
    }
    static func run(_ args: [String]) throws {
        precondition(args.count == 3 && ["tiny", "qwen"].contains(args[1]))
        let real = args[1] == "qwen"
        let payload = real ? Data(args[2].utf8) : try Data(contentsOf: URL(fileURLWithPath: args[2]))
        let client = try CPUClient(kind: real ? 1 : 0, payload: payload)
        let expected: [UInt32] = real ? [358, 2776, 264, 48948] : [41, 53, 41, 50]
        let promptCount: UInt32 = real ? 2 : 3
        precondition(client.waitReady { result in
            _ = checked(result, kind: UInt32(AIE_READY))
            // Oversized UTF-8 is rejected before tokenization/numerical writes,
            // after bounded off-main conversion; a valid request still works.
            precondition(client.begin(text: String(repeating: "é", count: 32_769), maxNew: 4) { invalid in
                dispatchPrecondition(condition: .onQueue(.main))
                guard case let .failure(error) = invalid else { fatalError("oversized input accepted") }
                precondition(error.code == UInt32(AIE_INVALID_INPUT))
                let completion: @Sendable (Result<CPUEvent, CPUFailure>) -> Void = { started in
                    let event = checked(started, kind: UInt32(AIE_STARTED))
                    precondition(event.processed == promptCount && event.history == promptCount)
                    advance(client, expected, promptCount, index: 0)
                }
                if real {
                    precondition(client.begin(text: "Hello! Hello! Hello! Hello!", maxNew: 4) { rejected in
                        guard case let .failure(error) = rejected else { fatalError("unqualified context admitted") }
                        precondition(error.code == UInt32(AIE_INVALID_INPUT))
                        precondition(client.begin(text: "Hello!", maxNew: 4, completion: completion))
                    })
                }
                else { precondition(client.begin(ids: [1, 5, 7], maxNew: 4, completion: completion)) }
            })
        })
        // Before Ready delivery, the Swift gate rejects extra work; no unbounded
        // closure queue and no synchronous numerical work on the main thread.
        precondition(!client.next { _ in fatalError("unadmitted callback") })
        dispatchMain()
    }
    static func advance(_ client: CPUClient, _ expected: [UInt32], _ prompt: UInt32, index: Int) {
        if index == expected.count {
            precondition(client.next { result in
                let stopped = checked(result, kind: UInt32(AIE_STOPPED))
                precondition(stopped.processed + 1 == stopped.history)
                precondition(client.reset { reset in
                    let event = checked(reset, kind: UInt32(AIE_RESET_EVENT))
                    precondition(event.processed == 0 && event.history == 0)
                    client.close()
                    print("Native parity/cancel/resume/reset passed; tokens=\(expected)")
                    exit(0)
                })
            })
            return
        }
        if index == 1 {
            precondition(client.setCancelled(true) == UInt32(AIE_OK))
            precondition(client.next { result in
                let event = checked(result, kind: UInt32(AIE_CANCEL_EVENT))
                precondition(event.processed == prompt && event.history == prompt + 1)
                precondition(client.setCancelled(false) == UInt32(AIE_OK))
                emit(client, expected, prompt, index: index)
            })
        } else { emit(client, expected, prompt, index: index) }
    }
    static func emit(_ client: CPUClient, _ expected: [UInt32], _ prompt: UInt32, index: Int) {
        precondition(client.next { result in
            let event = checked(result, kind: UInt32(AIE_TOKEN))
            precondition(event.token == expected[index])
            let expectedChunks = expected[0] == 358 ? [" I", "'m", " a", " beginner"] : ["41 ", "53 ", "41 ", "50 "]
            precondition(event.text == expectedChunks[index])
            precondition(event.processed == prompt + UInt32(index) && event.history == event.processed + 1)
            print("token=\(event.token!) processed=\(event.processed) history=\(event.history) text=\(String(reflecting: event.text))")
            advance(client, expected, prompt, index: index + 1)
        })
    }
}
