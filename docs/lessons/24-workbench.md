# 24 — One native workbench, one Rust engine

The Mac workbench has Search, Models, Chat and Diagnostics tabs. Search requests
public metadata only. Models separates the pinned source revision from the exact
execution bytes and shows qualification status. Chat invokes our CPU engine;
SwiftUI never implements transformer math, sampling, tokenization or KV storage.
The app has no automatic download or network inference. The install button opts
into the pinned source download; an already installed source works offline.

This is a minimal plain-prompt, single-turn UI, with explicit New chat/reset.
It does not imply a multi-turn instruction template, a capable assistant, long
context, phone support or ANE execution. Qwen requests admit at most8total
positions (prompt plus output), while cache allocation remainsC32. The Rust
worker enforces this before prefill. UI output limit is1–4. An over-budget prompt
is rejected whole and leaves history empty; no truncation or hidden rollover.
The synthetic offline demo takes comma-separated token IDs and admits C32.

Execution order:

1. Startup verifies bundled archive/executable digests in a detached task, then
   verifies installed source files. Only a matching, qualified build enables
   Qwen loading. Unknown or rebuilt execution bytes stay unqualified.
2. Load copies a bounded source descriptor into the Rust worker. The UI shows
   Loading and remains responsive; Ready arrives on the main queue. Unload
   invalidates the UI epoch, closes the handle and drops references immediately.
   Actual worker release is asynchronous; a subsequent load can temporarily be
   Busy. Closing the window also cancels requests/install and unloads.
3. Send submits bounded prompt input, records the visible turn and waits for
   Rust preflight/prefill. One response opens the next request; callbacks are
   main-queue guaranteed by CPUClient and enter MainActor under that invariant.
4. Each token event returns metadata and a bounded UTF-8 chunk **atomically**.
   Rust's established TextStream holds incomplete UTF-8 and suppresses special
   tokens. Tiny returns explicit IDs for the synthetic view. ABI2 adds
   `aie_poll_text`: caller grants disjoint Event/length/4096byte outputs; invalid
   descriptors or Pending preserve all outputs and consume no event. Queues
   retain one small owned delivery; Swift allocates one4KiB scratch per request,
   not per pending poll. Chunks are never truncated to fit the boundary.
5. Pause sets the independent atomic flag. A completed in-flight token may
   arrive, but no new request follows while paused. Empty-history cancellation
   can Resume back to Ready after clearing the flag; a nonempty prefix resumes
   the pending token step. Fatal errors show Failed and no valid evaluated prefix.
6. New chat explicitly resets Rust history/cache/text decoder and visible turn.
   It is disabled during active requests. Search/install cancellation targets
   the actual stored task; there is no delayed generic cancel that can kill a
   replacement request. Epochs also exclude stale publication after unload.

Concrete Qwen trace for plain `Hello!`: prompt IDs `[9707,0]` evaluate2positions.
ID358 emits chunk ` I` at2/3;2776 emits `'m` at3/4;264 emits ` a` at4/5;
48948 emits ` beginner` at5/6. Output budget stops with final token pending:
text ` I'm a beginner`, evaluated5/history6. Independent pinned tokenizer
full decoding matches that text; exact token/counter/cancel/reset checks run in
the native build. This short plain completion is not product-quality evidence.

Build using installed Xcode/Rust on the tested Mac26.5+:

```sh
apple/build-workbench.sh
```

The ordinary build runs tiny native parity and the offline UI-controller tests,
then leaves Qwen qualification false. To opt into already-local Qwen checks:

```sh
apple/build-workbench.sh \
  models/installed-qwen/7ae557604adf67be50417f59c2c2f167def9a775
```

Successful opt-in checks qualify the exact new archive/executable hashes in
bundle resources. Each newly built app starts with a provisional identity; failed qualification
cannot mark that new execution image as passed.
No model download/signing/account change occurs. Build outputs stay ignored at
`target/native/InferenceWorkbench.app`. Open that app normally in Finder. The
checked-in small artifact record is the reviewed current build; a future build
still requires its own qualification. This is a local Mac development app,
without a signed distribution/mobile target or persisted private prompts.

Controller regressions exercise cancel→immediate replacement, stale exclusion,
empty-history pause/resume, tiny generation/reset, rejected input and unload.
Rust tests cover atomic text-buffer preservation and pre-admission cancel/retry,
with existing UTF-8 split and lifecycle/failure tests retained. GUI accessibility
and screenshots check actual model controls, visible chat and counts. Large
network installation, mobile/backgrounding and energy tests remain unqualified.

Exercise: Pause arrives after the fourth Qwen token was emitted but before the
UI receives it. State which text/counts the UI may still display, and whether
Resume should evaluate that token when the output budget already ended. Do not
modify the sampler or cache to force an answer.
