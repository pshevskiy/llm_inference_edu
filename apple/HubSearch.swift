// Public metadata discovery only. Native networking is outside Rust inference.
import Foundation

struct HubModel: Decodable, Sendable, Identifiable {
    let id: String
    let author: String?
    let sha: String?
    let library_name: String?
    let pipeline_tag: String?
    let tags: [String]?
    let downloads: Int?
}
struct HubPage: Sendable {
    let models: [HubModel]
    let next: URL?
}
enum HubFailure: Error, Equatable {
    case invalidQuery, invalidResponse, invalidPageLink, oversizedResponse
    case http(Int, retryAfter: String?)
    case cancelled, noNextPage
}
struct HubQuery: Sendable {
    var text: String
    var publisher: String? = nil
    var library: String? = "mlx"
    var limit: Int = 20

    func url() throws -> URL {
        guard text.utf8.count <= 1_024, (1...50).contains(limit),
            [publisher, library].allSatisfy({ value in
                guard let value else { return true }
                return !value.isEmpty && value.utf8.count <= 128 &&
                    !value.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) })
            }), !text.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) })
        else { throw HubFailure.invalidQuery }
        var parts = URLComponents(string: "https://huggingface.co/api/models")!
        var items = [URLQueryItem(name: "search", value: text), URLQueryItem(name: "limit", value: String(limit))]
        if let publisher { items.append(URLQueryItem(name: "author", value: publisher)) }
        if let library { items.append(URLQueryItem(name: "filter", value: library)) }
        // Select small metadata; do not fetch model cards/config/files.
        for field in ["author", "sha", "library_name", "pipeline_tag", "tags", "downloads"] {
            items.append(URLQueryItem(name: "expand", value: field))
        }
        parts.queryItems = items
        guard let url = parts.url else { throw HubFailure.invalidQuery }
        return url
    }
}
struct HubResponse: Sendable {
    let status: Int
    let body: Data
    var link: String? = nil
    var retryAfter: String? = nil
}

private final class HubNoRedirect: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void) { completionHandler(nil) }
}

enum HubHTTP {
    static let maximumBytes = 1_048_576
    static func allowed(_ url: URL) -> Bool {
        guard let p = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return false }
        return p.scheme == "https" && p.host == "huggingface.co" && p.port == nil &&
            p.user == nil && p.password == nil && p.fragment == nil &&
            p.percentEncodedPath == "/api/models" && url.absoluteString.utf8.count <= 16_384
    }
    static func fetch(_ url: URL) async throws -> HubResponse {
        guard allowed(url) else { throw HubFailure.invalidPageLink }
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 15
        config.timeoutIntervalForResource = 30
        config.httpCookieStorage = nil
        config.urlCredentialStorage = nil
        let session = URLSession(configuration: config, delegate: HubNoRedirect(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("llm-inference-workbench/0.1", forHTTPHeaderField: "User-Agent")
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else { throw HubFailure.invalidResponse }
        let retry = http.value(forHTTPHeaderField: "Retry-After")
        guard http.statusCode == 200 else { throw HubFailure.http(http.statusCode, retryAfter: retry) }
        guard response.expectedContentLength <= Int64(maximumBytes) else { throw HubFailure.oversizedResponse }
        var body = Data()
        body.reserveCapacity(response.expectedContentLength > 0 ? Int(response.expectedContentLength) : 8_192)
        // Hard streamed cap also covers chunked/incorrect Content-Length.
        for try await byte in bytes {
            try Task.checkCancellation()
            guard body.count < maximumBytes else { throw HubFailure.oversizedResponse }
            body.append(byte)
        }
        return HubResponse(status: http.statusCode, body: body,
            link: http.value(forHTTPHeaderField: "Link"), retryAfter: retry)
    }
    static func nextLink(_ value: String?) throws -> URL? {
        guard let value, !value.isEmpty else { return nil }
        guard value.utf8.count <= 16_384 else { throw HubFailure.invalidPageLink }
        // HF emits <url>; rel="next". Restrict this small contract and reject
        // malformed/ambiguous headers; URLs themselves use Foundation's parser.
        let expression = try NSRegularExpression(pattern: #"(?:^|,)\s*<([^>]*)>\s*;\s*rel="([^"]*)"\s*(?=,|$)"#)
        let ns = value as NSString
        let matches = expression.matches(in: value, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { throw HubFailure.invalidPageLink }
        var next: URL?
        var covered = IndexSet()
        for match in matches {
            covered.insert(integersIn: match.range.location..<(match.range.location + match.range.length))
            let relation = ns.substring(with: match.range(at: 2)).split(separator: " ")
            if relation.contains("next") {
                guard next == nil, let url = URL(string: ns.substring(with: match.range(at: 1))), allowed(url)
                else { throw HubFailure.invalidPageLink }
                next = url
            }
        }
        // Do not silently ignore malformed sections beside a valid link.
        for i in 0..<ns.length where !covered.contains(i) {
            guard ns.character(at: i) == 44 || ns.character(at: i) == 32 else { throw HubFailure.invalidPageLink }
        }
        return next
    }
    static func page(_ response: HubResponse, limit: Int) throws -> HubPage {
        guard response.status == 200 else { throw HubFailure.http(response.status, retryAfter: response.retryAfter) }
        guard response.body.count <= maximumBytes else { throw HubFailure.oversizedResponse }
        let models: [HubModel]
        do { models = try JSONDecoder().decode([HubModel].self, from: response.body) }
        catch { throw HubFailure.invalidResponse }
        guard models.count <= limit, Set(models.map(\.id)).count == models.count,
            models.allSatisfy({ !$0.id.isEmpty && $0.id.utf8.count <= 256 }) else { throw HubFailure.invalidResponse }
        return HubPage(models: models, next: try nextLink(response.link))
    }
}

// One active page request. Actor serializes admission/cursor state; cancellation
// plus a generation ID excludes late results even if a transport ignores cancel.
actor HubSearch {
    typealias Transport = @Sendable (URL) async throws -> HubResponse
    private let transport: Transport
    private var active: Task<HubPage, Error>?
    private var generation: UInt64 = 0
    private var cursor: URL?
    private var limit = 20
    init(transport: @escaping Transport = HubHTTP.fetch) { self.transport = transport }

    func cancel() {
        generation &+= 1
        active?.cancel()
        active = nil
    }
    func search(_ query: HubQuery) async throws -> HubPage {
        cancel()
        cursor = nil
        limit = query.limit
        return try await fetch(query.url(), limit: query.limit)
    }
    func next() async throws -> HubPage {
        guard let cursor else { throw HubFailure.noNextPage }
        cancel()
        return try await fetch(cursor, limit: limit)
    }
    private func fetch(_ url: URL, limit: Int) async throws -> HubPage {
        let identity = generation
        let transport = self.transport
        let task = Task {
            let response = try await transport(url)
            try Task.checkCancellation()
            return try HubHTTP.page(response, limit: limit)
        }
        active = task
        do {
            let page = try await withTaskCancellationHandler(operation: { try await task.value }, onCancel: { task.cancel() })
            guard generation == identity, !Task.isCancelled else { throw HubFailure.cancelled }
            cursor = page.next
            active = nil
            return page
        } catch {
            guard generation == identity else { throw HubFailure.cancelled }
            active = nil
            if error is CancellationError || Task.isCancelled { throw HubFailure.cancelled }
            throw error
        }
    }
}
