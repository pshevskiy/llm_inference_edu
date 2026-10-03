import Dispatch
import Foundation
import LLMCPUBridge

public struct CPUEvent: Sendable {
    public let kind: UInt32
    public let token: UInt32?
    public let text: String
    public let processed: UInt32
    public let history: UInt32
}
public struct CPUFailure: Error, Sendable {
    public let code: UInt32
}

// The immutable integer handle is safe to use for cancellation/close from any
// thread. All Swift flags use lock; polling runs on queue; at most one admitted
// operation/result can await main-queue delivery. Rust checks stale handles.
// No foreign pointer, model, tokenizer, sampler or cache is stored in Swift.
public final class CPUClient: @unchecked Sendable {
    private let handle: UInt64
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "llm.cpu.delivery", qos: .userInitiated)
    private var busy = false
    private var ready = false
    private var closed = false

    public init(kind: UInt32, payload: Data) throws {
        guard aie_abi_version() == 2 else { throw CPUFailure(code: UInt32(AIE_INVALID_INPUT)) }
        var identifier: UInt64 = 0
        let code = payload.withUnsafeBytes { bytes in
            aie_open(kind, bytes.bindMemory(to: UInt8.self).baseAddress, bytes.count, &identifier)
        }
        guard code == UInt32(AIE_OK) else { throw CPUFailure(code: code) }
        handle = identifier
    }

    // false means the operation was not admitted; no completion is queued.
    // True promises exactly one completion on DispatchQueue.main, including
    // immediate native errors or closure/timeout while waiting for a response.
    @discardableResult
    private func request(loading: Bool = false,
        action: @escaping @Sendable (UInt64) -> UInt32,
        completion: @escaping @Sendable (Result<CPUEvent, CPUFailure>) -> Void) -> Bool {
        lock.lock()
        guard !closed && !busy && (loading ? !ready : ready) else { lock.unlock(); return false }
        busy = true
        lock.unlock()
        queue.async {
            dispatchPrecondition(condition: .notOnQueue(.main))
            var result: Result<CPUEvent, CPUFailure>
            let code = action(self.handle)
            if code != UInt32(AIE_OK) {
                result = .failure(CPUFailure(code: code))
            } else {
                let deadline = DispatchTime.now().uptimeNanoseconds + 60_000_000_000
                var text = [UInt8](repeating: 0, count: 4096)
                while true {
                    var event = AieEvent()
                    var count = 0
                    let status = text.withUnsafeMutableBufferPointer { buffer in
                        aie_poll_text(self.handle, &event, buffer.baseAddress, buffer.count, &count)
                    }
                    if status == UInt32(AIE_OK) {
                        if event.kind == UInt32(AIE_ERROR) { result = .failure(CPUFailure(code: event.code)) }
                        else { result = .success(CPUEvent(kind: event.kind,
                            token: event.token == UInt32.max ? nil : event.token,
                            text: String(decoding: text.prefix(count), as: UTF8.self),
                            processed: event.processed, history: event.history)) }
                        break
                    }
                    if status != UInt32(AIE_PENDING) { result = .failure(CPUFailure(code: status)); break }
                    if DispatchTime.now().uptimeNanoseconds >= deadline {
                        self.close()
                        result = .failure(CPUFailure(code: 100)) // bounded client timeout
                        break
                    }
                    Thread.sleep(forTimeInterval: 0.002)
                }
            }
            let delivered = result
            DispatchQueue.main.async {
                self.lock.lock()
                if case let .success(event) = delivered, event.kind == UInt32(AIE_READY) { self.ready = true }
                self.busy = false
                self.lock.unlock()
                completion(delivered)
            }
        }
        return true
    }

    @discardableResult
    public func waitReady(_ completion: @escaping @Sendable (Result<CPUEvent, CPUFailure>) -> Void) -> Bool {
        request(loading: true, action: { _ in UInt32(AIE_OK) }, completion: completion)
    }
    @discardableResult
    public func begin(ids: [UInt32], maxNew: UInt32,
        completion: @escaping @Sendable (Result<CPUEvent, CPUFailure>) -> Void) -> Bool {
        request(action: { handle in ids.withUnsafeBufferPointer { buffer in
            aie_begin_ids(handle, buffer.baseAddress, buffer.count, maxNew)
        } }, completion: completion)
    }
    @discardableResult
    public func begin(text: String, maxNew: UInt32,
        completion: @escaping @Sendable (Result<CPUEvent, CPUFailure>) -> Void) -> Bool {
        request(action: { handle in
            // Admission happens before any input conversion. Scan/copy only a
            // bounded prefix on the delivery queue; reject rather than truncate.
            let bytes = Array(text.utf8.prefix(65_537))
            guard !bytes.isEmpty && bytes.count <= 65_536 else { return UInt32(AIE_INVALID_INPUT) }
            return bytes.withUnsafeBufferPointer { buffer in
                aie_begin_text(handle, buffer.baseAddress, buffer.count, maxNew)
            }
        }, completion: completion)
    }
    @discardableResult
    public func next(_ completion: @escaping @Sendable (Result<CPUEvent, CPUFailure>) -> Void) -> Bool {
        request(action: aie_next, completion: completion)
    }
    @discardableResult
    public func reset(_ completion: @escaping @Sendable (Result<CPUEvent, CPUFailure>) -> Void) -> Bool {
        request(action: aie_reset, completion: completion)
    }
    public func setCancelled(_ enabled: Bool) -> UInt32 {
        aie_set_cancel(handle, enabled ? 1 : 0)
    }
    public func close() {
        lock.lock()
        if closed { lock.unlock(); return }
        closed = true
        lock.unlock()
        _ = aie_set_cancel(handle, 1)
        _ = aie_close(handle)
    }
    deinit { _ = aie_close(handle) }
}
