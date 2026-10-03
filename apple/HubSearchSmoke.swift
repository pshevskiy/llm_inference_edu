// Offline contract tests; --live performs a separately labeled public API check.
import Foundation

private actor Scripts {
    var urls: [URL] = []
    var responses: [HubResponse]
    init(_ responses: [HubResponse]) { self.responses = responses }
    func take(_ url: URL) throws -> HubResponse {
        urls.append(url)
        guard !responses.isEmpty else { throw HubFailure.invalidResponse }
        return responses.removeFirst()
    }
}
private actor Held {
    var entered = false
    private var waiter: CheckedContinuation<Void, Never>?
    func take(_ url: URL) async -> HubResponse {
        entered = true
        await withCheckedContinuation { waiter = $0 }
        return HubResponse(status: 200, body: Data(#"[{"id":"old/model"}]"#.utf8))
    }
    func release() { waiter?.resume(); waiter = nil }
}
private actor Replaced {
    var entered = false
    var urls: [URL] = []
    private var waiter: CheckedContinuation<Void, Never>?
    func take(_ url: URL) async -> HubResponse {
        urls.append(url)
        let query = URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems!
        if query.contains(where: { $0.name == "search" && $0.value == "old" }) {
            entered = true
            await withCheckedContinuation { waiter = $0 }
            return HubResponse(status: 200, body: Data("[]".utf8))
        }
        if query.contains(where: { $0.name == "cursor" }) {
            return HubResponse(status: 200, body: Data("[]".utf8))
        }
        return HubResponse(status: 200, body: Data(#"[{"id":"new/model"}]"#.utf8),
            link: #"<https://huggingface.co/api/models?cursor=latest&limit=2>; rel="next""#)
    }
    func release() { waiter?.resume(); waiter = nil }
}
@main
struct HubSearchSmoke {
    static func failure(_ expected: HubFailure, _ action: () throws -> Void) {
        do { try action(); preconditionFailure("expected \(expected)") }
        catch { precondition(error as? HubFailure == expected, "unexpected \(error)") }
    }
    static func main() async throws {
        if CommandLine.arguments.contains("--live") {
            let search = HubSearch()
            let first = try await search.search(HubQuery(text: "Qwen", limit: 2))
            precondition(first.models.count <= 2)
            print("Live first page: \(first.models.map(\.id)); next=\(first.next != nil)")
            if first.next != nil {
                let second = try await search.next()
                print("Live next page: \(second.models.map(\.id))")
            }
            return
        }
        let url = try HubQuery(text: "a & b/é", publisher: "some-owner", library: nil, limit: 2).url()
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems!
        precondition(items.first { $0.name == "search" }!.value == "a & b/é")
        precondition(!items.contains { $0.name == "filter" })
        precondition(items.first { $0.name == "author" }!.value == "some-owner")
        let defaultURL = try HubQuery(text: "").url()
        precondition(defaultURL.absoluteString.contains("filter=mlx"))
        failure(.invalidQuery) { _ = try HubQuery(text: "x", limit: 51).url() }
        failure(.invalidQuery) { _ = try HubQuery(text: String(repeating: "x", count: 1_025)).url() }
        failure(.invalidQuery) { _ = try HubQuery(text: "x", publisher: "").url() }
        failure(.invalidQuery) { _ = try HubQuery(text: "x\n").url() }

        let link = #"<https://huggingface.co/api/models?cursor=a,b&limit=2>; rel="next", <https://huggingface.co/api/models?cursor=z>; rel="last""#
        let parsedLink = try HubHTTP.nextLink(link)
        precondition(parsedLink?.absoluteString.contains("cursor=a,b") == true)
        for bad in [#"<https://evil.test/api/models>; rel="next""#,
            #"<http://huggingface.co/api/models>; rel="next""#,
            #"<https://huggingface.co/api/models/else>; rel="next""#,
            #"<https://u@huggingface.co/api/models>; rel="next""#,
            #"<https://huggingface.co/api/models#fragment>; rel="next""#,
            link + #", <https://huggingface.co/api/models?cursor=x>; rel="next""#,
            link + ", garbage"] {
            failure(.invalidPageLink) { _ = try HubHTTP.nextLink(bad) }
        }
        let missing = HubResponse(status: 200, body: Data(#"[{"id":"other-owner/model"}]"#.utf8), link: link)
        let page = try HubHTTP.page(missing, limit: 2)
        precondition(page.models[0].library_name == nil && page.models[0].sha == nil && page.models[0].tags == nil)
        precondition(page.models[0].author == nil)
        failure(.invalidResponse) { _ = try HubHTTP.page(HubResponse(status: 200, body: Data("{}".utf8)), limit: 2) }
        failure(.invalidResponse) { _ = try HubHTTP.page(missing, limit: 0) }
        failure(.invalidResponse) {
            _ = try HubHTTP.page(HubResponse(status: 200, body: Data(#"[{"id":"a"},{"id":"a"}]"#.utf8)), limit: 2)
        }
        failure(.oversizedResponse) {
            _ = try HubHTTP.page(HubResponse(status: 200, body: Data(repeating: 32, count: HubHTTP.maximumBytes+1)), limit: 2)
        }
        failure(.http(429, retryAfter: "5")) {
            _ = try HubHTTP.page(HubResponse(status: 429, body: Data(), retryAfter: "5"), limit: 2)
        }
        let scripts = Scripts([missing, HubResponse(status: 200, body: Data("[]".utf8))])
        let search = HubSearch(transport: { try await scripts.take($0) })
        _ = try await search.search(HubQuery(text: "a", limit: 2))
        let second = try await search.next()
        precondition(second.models.isEmpty && second.next == nil)
        let recorded = await scripts.urls
        precondition(recorded.count == 2)
        do { _ = try await search.next(); preconditionFailure("no next expected") }
        catch { precondition(error as? HubFailure == .noNextPage) }

        let held = Held()
        let cancellable = HubSearch(transport: { await held.take($0) })
        let old = Task { try await cancellable.search(HubQuery(text: "old")) }
        while !(await held.entered) { await Task.yield() }
        await cancellable.cancel()
        await held.release() // A hostile/late transport ignores cancellation.
        do { _ = try await old.value; preconditionFailure("stale response published") }
        catch { precondition(error as? HubFailure == .cancelled) }
        let replacement = Replaced()
        let latest = HubSearch(transport: { await replacement.take($0) })
        let stale = Task { try await latest.search(HubQuery(text: "old", limit: 2)) }
        while !(await replacement.entered) { await Task.yield() }
        let newer = try await latest.search(HubQuery(text: "new", limit: 2))
        precondition(newer.models[0].id == "new/model")
        await replacement.release()
        do { _ = try await stale.value; preconditionFailure("replaced response published") }
        catch { precondition(error as? HubFailure == .cancelled) }
        _ = try await latest.next()
        let replacementURLs = await replacement.urls
        precondition(replacementURLs.count == 3 && replacementURLs[2].absoluteString.contains("cursor=latest"))

        let callerHeld = Held()
        let callerClient = HubSearch(transport: { await callerHeld.take($0) })
        let caller = Task { try await callerClient.search(HubQuery(text: "caller")) }
        while !(await callerHeld.entered) { await Task.yield() }
        caller.cancel()
        await callerHeld.release()
        do { _ = try await caller.value; preconditionFailure("caller cancellation ignored") }
        catch { precondition(error as? HubFailure == .cancelled) }
        print("Offline Hub query/metadata/pagination/bounds/errors/cancellation contracts passed")
    }
}
