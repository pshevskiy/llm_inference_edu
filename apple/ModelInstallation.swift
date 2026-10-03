import CryptoKit
import Foundation

struct ModelAsset: Codable, Sendable, Equatable { let bytes: UInt64; let sha256: String }
struct SourceInventory: Codable, Sendable, Equatable {
    let repository: String
    let revision: String
    let license: String
    let format: String
    let shards: [String]
    let assets: [String: ModelAsset]
}
struct CPUArtifact: Codable, Sendable, Equatable {
    let id: String
    let librarySHA256: String
    let executableSHA256: String
    let validation: String
}
struct ModelIdentity: Sendable {
    let repository: String
    let revision: String?
}
enum Compatibility: String, Sendable { case runnable, recipeAvailable, discoveryOnly, unsupported }
struct ModelCompatibility: Sendable {
    let discovered: ModelIdentity
    let source: ModelIdentity?
    let execution: CPUArtifact?
    let status: Compatibility
    let explanation: String
}
struct CPURecipe: Sendable {
    let source: SourceInventory
    let artifact: CPUArtifact
    // Inventory and artifact are supplied by the trusted application bundle,
    // never by a discovered repository's Python/Jinja/config code.
    func compatibility(_ discovered: HubModel, installation: VerifiedInstallation? = nil, execution: VerifiedExecution? = nil) -> ModelCompatibility {
        let identity = ModelIdentity(repository: discovered.id, revision: discovered.sha)
        guard discovered.id == source.repository else {
            return ModelCompatibility(discovered: identity, source: nil, execution: nil,
                status: .discoveryOnly, explanation: "No qualified execution recipe for this repository. MLX metadata is discovery only.")
        }
        guard discovered.sha == source.revision else {
            return ModelCompatibility(discovered: identity, source: nil, execution: nil,
                status: discovered.sha == nil ? .discoveryOnly : .unsupported,
                explanation: "Only the separately pinned source revision is qualified; this revision is unknown or different.")
        }
        return ModelCompatibility(discovered: identity,
            source: ModelIdentity(repository: source.repository, revision: source.revision), execution: artifact,
            status: installation?.matches(self) == true && execution?.artifact == artifact ? .runnable : .recipeAvailable,
            explanation: "Rust FP32 CPU on the qualified Mac; C32 allocated, only short ≤8-position real-model checks qualified. ANE and iPhone unqualified.")
    }
    func validate() throws -> UInt64 {
        guard source.repository == "Qwen/Qwen2.5-0.5B-Instruct",
            source.revision == "7ae557604adf67be50417f59c2c2f167def9a775",
            source.format == "safetensors", source.shards == ["model.safetensors"],
            Set(source.assets.keys) == Set(["LICENSE", "README.md", "config.json", "generation_config.json",
                "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "model.safetensors"]),
            artifact.id == "rust-cpu-fp32-c32-v1", Self.digestValid(artifact.librarySHA256), Self.digestValid(artifact.executableSHA256)
        else { throw InstallFailure.invalidRecipe }
        var total: UInt64 = 0
        for (name, asset) in source.assets {
            guard Self.filenameValid(name), asset.bytes > 0, Self.digestValid(asset.sha256) else { throw InstallFailure.invalidRecipe }
            let sum = total.addingReportingOverflow(asset.bytes)
            guard !sum.overflow else { throw InstallFailure.invalidRecipe }
            total = sum.partialValue
        }
        return total
    }
    static func filenameValid(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".." && value.utf8.allSatisfy {
            (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0) || [45,46,95].contains($0)
        }
    }
    static func digestValid(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
}
struct VerifiedInstallation: Sendable {
    let directory: URL
    fileprivate let source: SourceInventory
    fileprivate let artifact: CPUArtifact
    fileprivate init(directory: URL, recipe: CPURecipe) {
        self.directory = directory; source = recipe.source; artifact = recipe.artifact
    }
    fileprivate func matches(_ recipe: CPURecipe) -> Bool {
        source == recipe.source && artifact == recipe.artifact
    }
}
struct VerifiedExecution: Sendable {
    let executable: URL
    fileprivate let artifact: CPUArtifact
    fileprivate init(executable: URL, artifact: CPUArtifact) { self.executable = executable; self.artifact = artifact }
}
enum InstallFailure: Error, Equatable {
    case busy, invalidRecipe, insufficientSpace, invalidDirectory, integrity(String), http(Int), cancelled
}

// One bounded download. URLSession writes to disk instead of retaining weights
// in Data; delegates cancel on excess bytes and move the temporary file while
// its lifetime is still guaranteed. No repository code is loaded or executed.
private final class AssetTransfer: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let expected: UInt64
    private let destination: URL
    private var continuation: CheckedContinuation<Void, Error>?
    private var task: URLSessionDownloadTask?
    private var cancelled = false
    private var session: URLSession?
    init(expected: UInt64, destination: URL) { self.expected = expected; self.destination = destination }
    static func allowed(_ url: URL) -> Bool {
        guard let p = URLComponents(url: url, resolvingAgainstBaseURL: false), let host = p.host else { return false }
        return p.scheme == "https" && p.port == nil && p.user == nil && p.password == nil && p.fragment == nil &&
            (host == "huggingface.co" || host.hasSuffix(".huggingface.co") || host.hasSuffix(".hf.co"))
    }
    func run(_ url: URL) async throws {
        guard Self.allowed(url) else { throw InstallFailure.invalidRecipe }
        try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                lock.lock()
                if cancelled { lock.unlock(); c.resume(throwing: InstallFailure.cancelled); return }
                continuation = c
                let config = URLSessionConfiguration.ephemeral
                config.httpCookieStorage = nil; config.urlCredentialStorage = nil
                config.timeoutIntervalForRequest = 30; config.timeoutIntervalForResource = 300
                let queue = OperationQueue(); queue.maxConcurrentOperationCount = 1
                let s = URLSession(configuration: config, delegate: self, delegateQueue: queue)
                let t = s.downloadTask(with: url)
                session = s; task = t
                lock.unlock()
                t.resume()
            }
        }, onCancel: { self.cancel() })
    }
    func cancel() {
        lock.lock(); cancelled = true; let t = task; lock.unlock()
        t?.cancel()
    }
    private func complete(_ result: Result<Void, Error>) {
        lock.lock()
        let c = continuation; continuation = nil
        let s = session; session = nil; task = nil
        lock.unlock()
        s?.invalidateAndCancel()
        c?.resume(with: result)
    }
    func urlSession(_ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(request.url.map(Self.allowed) == true ? request : nil)
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        if totalBytesWritten < 0 || UInt64(totalBytesWritten) > expected ||
            (totalBytesExpectedToWrite >= 0 && UInt64(totalBytesExpectedToWrite) > expected) {
            downloadTask.cancel()
            complete(.failure(InstallFailure.integrity(destination.lastPathComponent)))
        }
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        do {
            guard let response = downloadTask.response as? HTTPURLResponse else { throw InstallFailure.http(0) }
            guard response.statusCode == 200 else { throw InstallFailure.http(response.statusCode) }
            let size = try FileManager.default.attributesOfItem(atPath: location.path)[.size] as? NSNumber
            guard size?.uint64Value == expected else { throw InstallFailure.integrity(destination.lastPathComponent) }
            lock.lock(); let stopped = cancelled; lock.unlock()
            guard !stopped else { throw InstallFailure.cancelled }
            try FileManager.default.moveItem(at: location, to: destination)
            complete(.success(()))
        } catch { complete(.failure(error)) }
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            lock.lock(); let stopped = cancelled; lock.unlock()
            complete(.failure(stopped ? InstallFailure.cancelled : error))
        }
    }
}

actor ModelInstaller {
    typealias Fetch = @Sendable (URL, URL, ModelAsset) async throws -> Void
    private var active: Task<VerifiedInstallation, Error>?
    static func download(_ source: URL, _ destination: URL, _ asset: ModelAsset) async throws {
        try await AssetTransfer(expected: asset.bytes, destination: destination).run(source)
    }
    func cancel() { active?.cancel() }
    func install(_ recipe: CPURecipe, root: URL, availableBytes: UInt64? = nil,
        fetch: @escaping Fetch = ModelInstaller.download) async throws -> VerifiedInstallation {
        guard active == nil else { throw InstallFailure.busy }
        let task = Task.detached(priority: .userInitiated) {
            try await Self.perform(recipe, root: root, availableBytes: availableBytes, fetch: fetch)
        }
        active = task
        defer { active = nil }
        do {
            return try await withTaskCancellationHandler(operation: { try await task.value }, onCancel: { task.cancel() })
        } catch {
            if error is CancellationError || Task.isCancelled { throw InstallFailure.cancelled }
            throw error
        }
    }
    static func hash(_ url: URL, asset: ModelAsset) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true,
            let size = values.fileSize, UInt64(size) == asset.bytes else { return false }
        let input = try FileHandle(forReadingFrom: url); defer { try? input.close() }
        var digest = SHA256()
        while let chunk = try input.read(upToCount: 4 * 1024 * 1024), !chunk.isEmpty {
            try Task.checkCancellation(); digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined() == asset.sha256
    }
    static func verifyExecution(_ artifact: CPUArtifact, library: URL, executable: URL) throws -> VerifiedExecution {
        guard FileManager.default.isExecutableFile(atPath: executable.path),
            CPURecipe.digestValid(artifact.librarySHA256), CPURecipe.digestValid(artifact.executableSHA256)
        else { throw InstallFailure.integrity("execution artifact") }
        for (url, expected) in [(library, artifact.librarySHA256), (executable, artifact.executableSHA256)] {
            let v = try url.resourceValues(forKeys: [.fileSizeKey])
            guard let size = v.fileSize, size > 0,
                try hash(url, asset: ModelAsset(bytes: UInt64(size), sha256: expected))
            else { throw InstallFailure.integrity("execution artifact") }
        }
        return VerifiedExecution(executable: executable, artifact: artifact)
    }
    static func verify(_ recipe: CPURecipe, directory: URL) throws -> VerifiedInstallation {
        _ = try recipe.validate()
        let v = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard v.isDirectory == true, v.isSymbolicLink != true else { throw InstallFailure.invalidDirectory }
        for (name, asset) in recipe.source.assets.sorted(by: { $0.key < $1.key }) {
            try Task.checkCancellation()
            guard try hash(directory.appendingPathComponent(name), asset: asset) else { throw InstallFailure.integrity(name) }
        }
        return VerifiedInstallation(directory: directory, recipe: recipe)
    }
    private static func perform(_ recipe: CPURecipe, root: URL, availableBytes: UInt64?, fetch: Fetch) async throws -> VerifiedInstallation {
        let total = try recipe.validate()
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let values = try root.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey, .volumeAvailableCapacityKey])
        guard values.isDirectory == true, values.isSymbolicLink != true else { throw InstallFailure.invalidDirectory }
        let final = root.appendingPathComponent(recipe.source.revision, isDirectory: true)
        if fm.fileExists(atPath: final.path) { return try verify(recipe, directory: final) }
        let twice = total.multipliedReportingOverflow(by: 2)
        let budget = twice.partialValue.addingReportingOverflow(64 * 1024 * 1024)
        guard !twice.overflow, !budget.overflow else { throw InstallFailure.invalidRecipe }
        let measured = values.volumeAvailableCapacity.map { UInt64(max(0, $0)) }
        guard let free = availableBytes ?? measured, free >= budget.partialValue else { throw InstallFailure.insufficientSpace }
        let stage = root.appendingPathComponent(".install-\(UUID().uuidString).partial", isDirectory: true)
        try fm.createDirectory(at: stage, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        defer { try? fm.removeItem(at: stage) }
        for (name, asset) in recipe.source.assets.sorted(by: { $0.key < $1.key }) {
            try Task.checkCancellation()
            let source = URL(string: "https://huggingface.co/\(recipe.source.repository)/resolve/\(recipe.source.revision)/\(name)")!
            let destination = stage.appendingPathComponent(name)
            try await fetch(source, destination, asset)
            guard try hash(destination, asset: asset) else { throw InstallFailure.integrity(name) }
        }
        try Task.checkCancellation()
        // Destination must not exist. Atomic directory rename is the single
        // publication point; crashes may leave a partial stage, never installed.
        try fm.moveItem(at: stage, to: final)
        return VerifiedInstallation(directory: final, recipe: recipe)
    }
}
