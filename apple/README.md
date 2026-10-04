# Native CPU boundary

`CPUClient.swift` is a thin Swift 6 asynchronous driver for our Rust engine.
`BoundarySmoke.swift` is a development parity/lifecycle executable, not an app.
The C header documents ABI ownership. A concrete static-library wrapper isolates
FFI unsafe code from the numerical package; it adds no third-party dependencies.

On the tested arm64 Mac, using installed Xcode without changing the global
Command Line Tools selection:

```sh
cargo build --offline --release -p llm_cpu_bridge
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swiftc \
  -target arm64-apple-macosx26.5 -swift-version 6 -warnings-as-errors -O -g \
  -parse-as-library -I apple/cpu_bridge/include \
  apple/CPUClient.swift apple/BoundarySmoke.swift \
  target/release/libllm_cpu_bridge.a -o /tmp/llm-native-smoke
/tmp/llm-native-smoke tiny fixtures/tiny-model.json
```

Optional already-downloaded model check (no download performed by this command):

```sh
/tmp/llm-native-smoke qwen \
  models/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775
```

Expected tiny IDs `[41,53,41,50]`; Qwen plain `Hello!` IDs
`[358,2776,264,48948]`. Checks include main-queue delivery, request backpressure,
oversized UTF-8 rejection followed by valid admission, cancellation/resume,
reset, and pending-token counters. Rust portable tests additionally check
invalid memory descriptors, unchanged output on pre-admission failures,
terminal numerical failure/panic, stale handles, and close during active work.
These Mac checks do not validate an iPhone, signing, suspension or memory budget.

See [the lesson](../docs/lessons/20-native-boundary.md) for execution order and
[the plan](../docs/PLAN.md) for evidence and remaining gates.

## Public model discovery

`HubSearch.swift` searches public Hub metadata with optional publisher/library
filters, one bounded page per request and cancellation. It does not download,
execute or certify discovered models. `HubSearchSmoke.swift` contains offline
contract checks and an optional `--live` two-page metadata check; see
[lesson22](../docs/lessons/22-discovery.md) for commands and exact limits.

## Pinned installation and compatibility

`ModelInstallation.swift` separates repository/source/execution identities and
installs only the trusted pinned CPU source recipe. It checks disk space,
stages on disk, verifies every size/hash and atomically publishes a complete
directory. Cancellation cleans up only its own stage. Runnable needs source and
actual execution-byte verification; MLX search results remain discovery-only.
`InstallationSmoke.swift` has small offline fixtures, optional LICENSE-only
`--network-small`, and an opt-in `--local` existing-source check tied to the
recorded Mac artifacts. See [lesson23](../docs/lessons/23-installation.md).

## Mac workbench

Build `apple/build-workbench.sh`; the app appears at
`target/native/InferenceWorkbench.app`. Optional first argument supplies an
already-local pinned Qwen directory and enables real native qualification;
ordinary builds qualify only the tiny demo. Both run offline UI-controller
regressions. Open the app in Finder for search, explicit installation/load,
short plain-prompt chat, pause/resume, unload/reset and diagnostics. Full build
contract, eight-position admission and artifact verification are in
[lesson24](../docs/lessons/24-workbench.md). No iOS/signing claim.

## iPhone development slice (T21)

The tiny-model SwiftUI project and staged simulator/device instructions are in
[iphone/README.md](iphone/README.md). Swift SDK checks and Mac controller tests
pass; platform app builds and both Simulator UI tests pass after approved setup.
The Simulator also passes 107 engine and 8 bridge Rust tests. Physical-phone
validation and its resource budget remain pending.
