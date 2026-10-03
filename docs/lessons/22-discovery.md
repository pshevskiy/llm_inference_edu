# 22 — Discover repositories without promising execution

Native Foundation networking handles public Hub metadata; our Rust numerical
engine is unchanged. `HubSearch` is one concrete actor, not a backend hierarchy.
`HubQuery` creates `/api/models` query items with text search, an optional author
and optional library/tag filter. MLX is the default filter, but removing it
allows general discovery. No publisher is assumed to own the entire catalog.
Foundation encodes query values; `a & b/é` cannot become an extra HTTP parameter.

The current public Hub client documents filter/author/search/limit/expand:
[official list_models implementation](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py).
Its [pagination helper](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_pagination.py)
follows HTTP Link's next relation. We verified the actual endpoint and next
header. The downloaded current OpenAPI contains other model routes but omits
list-models; it is not claimed as its complete schema.

Execution order:

1. A query validates short fields and page limit1–50 (default20). It selects only
   author, revision, library, task, tags and download metadata, avoiding large
   cards, configs and file listings. Missing values stay nil/unknown.
2. Search cancels the previous request and clears its cursor. One task fetches
   an ephemeral, unauthenticated page, with15s request/30s resource deadlines.
   Cookies/credential storage are disabled; redirects are refused.
3. Response bytes stream into a capped1MiB buffer. Both declared length and
   actual streamed length are checked, then JSON and row limits are validated.
   Non200 responses retain their status and Retry-After; there is no automatic
   retry flood. Network/JSON errors surface to the caller.
4. Link parsing accepts HF's observed `<url>; rel="next"`/`"last"` grammar,
   including commas inside an angle-delimited URL. It rejects malformed or
   duplicate next relations. This is a deliberately strict supported contract,
   not a general RFC Link parser. Cursor URLs must remain HTTPS on the exact
   public Hub `/api/models` path, without credentials, fragment or custom port.
5. The request's generation number must still be current before publishing a
   response or replacing the cursor. Cancellation is checked even when an old
   transport ignores it. Caller task cancellation also cancels the child fetch.
6. Next requests use the server cursor. Results are returned one page at a time,
   rather than downloading/retaining the whole catalog. An empty page is valid;
   absent next means pagination ended. A failed next can be retried from the
   previously valid cursor.

Trace: hold query A's response. Query B cancels A and completes with cursor B.
A then arrives late; A returns Cancelled and cannot clear B's active request or
replace B's cursor. Next therefore follows B. Offline tests force that ordering
with continuations rather than relying on timing sleeps.

A repository ID/revision in these results is discovery metadata, not an installed
source or qualified execution artifact. Library/tags can be missing or wrong;
MLX tagging does not certify our Rust loader, conversion, quality or ANE. T23
will provide the compatibility/installation decisions separately.

Build and run the network-free contract checks on installed Swift6:

```sh
xcrun swiftc -swift-version 6 -warnings-as-errors -O -parse-as-library \
  apple/HubSearch.swift apple/HubSearchSmoke.swift -o /tmp/llm-hub-search
/tmp/llm-hub-search
```

An optional public two-page metadata check (no authentication/model download):

```sh
/tmp/llm-hub-search --live
```

The native checks use the installed Mac Swift SDK; portable Rust numerical tests
remain independent of network/Apple hardware. No mobile result is claimed.

Exercise: query A returns a valid next cursor, then a new query B fails with
HTTP429. Decide whether the UI should offer A's next page and explain which
query's results it would display. Do not implement a retry policy yet.
