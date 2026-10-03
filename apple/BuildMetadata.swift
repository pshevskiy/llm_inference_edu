import CryptoKit
import Foundation
@main
struct BuildMetadata {
    static func hash(_ file: URL) throws -> String {
        let input = try FileHandle(forReadingFrom: file); defer { try? input.close() }
        var digest = SHA256()
        while let bytes = try input.read(upToCount: 4 * 1024 * 1024), !bytes.isEmpty { digest.update(data: bytes) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
    static func main() throws {
        let args = CommandLine.arguments
        precondition(args.count == 5)
        let resources = URL(fileURLWithPath: args[1], isDirectory: true)
        let qualified = args[4] == "1"
        let record: [String: Any] = ["id": "rust-cpu-fp32-c32-v1", "librarySHA256": try hash(URL(fileURLWithPath: args[2])),
            "executableSHA256": try hash(URL(fileURLWithPath: args[3])), "qwenQualified": qualified,
            "validation": qualified ? "Mac native tiny/Qwen parity and lifecycle smoke passed for this build; Rust numerical engine retains short ≤8-position FP32 qualification. ANE/iPhone unqualified." : "Tiny Mac native parity/lifecycle smoke only. This build's Qwen execution qualification is pending. ANE/iPhone unqualified."]
        try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys, .prettyPrinted])
            .write(to: resources.appendingPathComponent("cpu-native-artifact.json"), options: .atomic)
    }
}
