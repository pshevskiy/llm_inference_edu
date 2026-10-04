# Inference Lab for iPhone — T21 development slice

A small SwiftUI host for the existing Rust CPU engine. It bundles only the
435,741-byte synthetic model, prepares `[1,5,7]`, and requests four greedy tokens
one at a time. It exposes evaluated-prefix and history counts, pause/resume,
reset and unload. There is no pretrained-model installer or conversational
assistant in this slice. No new Rust dependencies or parallel inference engine.

**Current validation (2026-10-04):** the app runs in the iPhone 14 Simulator
on the owner's M1 Pro. Both UI tests and 115 Rust engine/bridge tests pass there.
Host formatting, Clippy and all 117 workspace tests pass. Release Simulator and
unsigned physical-device builds pass. Rust iOS targets, Xcode first-launch
components and the arm64 iOS 27.0 runtime were installed after owner approval.
No physical phone is connected. T21 remains unchecked; physical parity,
real-model admission, memory/thermal qualification and owner review are pending.

## Simulator first

Open `apple/iphone/InferencePhone.xcodeproj` in Xcode. Complete Xcode's first-launch
component setup and install an iOS Simulator runtime through Xcode Settings →
Components. These steps can require administrator access and several GB.
They are not performed automatically. Official instructions:
[Apple Xcode components](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components).

After explicit approval, add components to the **existing** Rust 1.90 toolchain:

```sh
rustup target add --toolchain 1.90 aarch64-apple-ios-sim aarch64-apple-ios
```

The project runs `build-rust.sh`: it uses Rust 1.90 explicitly, checks the target
before building offline, and writes separate simulator/device static archives.
Both are arm64; they cannot be substituted for one another or the Mac archive.
Cargo's build scripts/procedural macros execute on the Mac: the script supplies
a Mac SDK environment default for those host tools. Rust and native target
components resolve the explicit iOS SDK separately.
Only arm64 builds are supported in this slice. No global developer-directory,
account, team, provisioning, or signing settings are changed by our scripts.

From the repository root, unsigned simulator build check:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild \
  -project apple/iphone/InferencePhone.xcodeproj -scheme InferencePhone \
  -configuration Release -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath target/iphone-xcode CODE_SIGNING_ALLOWED=NO build
```

List available simulators using scoped `DEVELOPER_DIR` and `xcrun simctl list
devices available`; choose a real listed UUID. Do not invent a device/runtime.
Run the checked-in UI tests (replace `SIMULATOR_UUID`; use a fresh result path):

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild \
  -project apple/iphone/InferencePhone.xcodeproj -scheme InferencePhone \
  -configuration Debug -destination 'platform=iOS Simulator,id=SIMULATOR_UUID' \
  -derivedDataPath target/iphone-xcode \
  -resultBundlePath docs/evidence/t21-simulator.xcresult \
  CODE_SIGNING_ALLOWED=NO test
```

The two tests passed on `Inference Lab iPhone 14`, UUID
`66CC7053-4EFC-4007-8A25-9FF7450CB411`, arm64 iOS 27.0/build24A434,
using Xcode27.0/build27A266a on M1Pro16GiB/macOS26.6.2/build25G83.
They check `[41,53,41,50]`, exact counters, Home/foreground pause, explicit
resume, finished-state preservation, reset, unload, empty-session foreground
reload, and notification-driven closure/reload. The initial test exposed a
disabled Load button: activity must be observable and scene phase must come
from the window's view. The corrected tests pass without changing expectations.

Xcode 27 shows simulated devices in **Xcode → Open Developer Tool → Device Hub**
([Apple Device Hub](https://developer.apple.com/documentation/xcode/device-hub)).
The Debug app includes a **Memory-warning demo** button. It posts the public
UIKit notification to exercise the app's listener; the button is absent from
Release. This is synthetic notification delivery, not OS memory exhaustion,
jetsam or physical pressure qualification. Manual inspection also confirmed
`41` with visibly rendered `Evaluated: 3 · History: 4`, then notification-driven
Unloaded state. Logs, result bundles and screenshots are ignored under
`docs/evidence/t21-*`; evidence is local rather than bundled in Git.

Run the existing numerical and bridge tests inside an explicitly booted
Simulator, with no downloads or auto-created devices (repeat for
`llm_cpu_bridge` instead of `llm_inference`):

```sh
task_rustc=$(rustup which --toolchain 1.90 rustc)
task_cargo=$(rustup which --toolchain 1.90 cargo)
task_sdk=$(DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun --sdk macosx --show-sdk-path)
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer SDKROOT="$task_sdk" \
  RUSTC="$task_rustc" CARGO_TARGET_DIR="$PWD/target/iphone-rust" \
  INFERENCE_SIMULATOR_UUID=SIMULATOR_UUID \
  CARGO_TARGET_AARCH64_APPLE_IOS_SIM_RUNNER="/bin/sh $PWD/apple/iphone/run-simulator-test.sh" \
  "$task_cargo" test --offline --locked --release -p llm_inference \
    --target aarch64-apple-ios-sim --tests -- --test-threads=1
```

Actual results: 107 engine tests plus 8 bridge tests pass. These cover independent
operator/intermediate/logit references and real Rust failure/cancellation/cache
contracts on iOS Simulator. They establish portability, not phone throughput,
physical GPU/ANE execution or an admission budget.

## Mac controller check

```sh
/bin/sh apple/iphone/check-controller.sh
```

This builds the Mac Rust bridge and controller with Swift 6/warnings as errors.
It tests independent expected tokens/counters, duplicate-request rejection,
pause during load and requests, background/foreground during prompt/reset,
memory-warning handler closure, stale callbacks, reload and missing-resource
failure, and synthetic terminal-failure display invalidation. The failure hook
is compiled only into this Mac check, not the shipped app; it tests the controller
transition rather than injecting a Rust fault. It does **not** simulate iOS
lifecycle delivery or phone memory pressure.

## Physical iPhone after simulator validation

Connect/unlock/trust the iPhone 14, enable Developer Mode, and check Xcode 27
Device Hub. Older Xcode versions use Window → Devices and Simulators.
`devicectl list devices` can include simulated devices: check the Reality column.
The final listing here contains only `simulated`, so physical checks are **not run**.
The owner selects their signing team in Xcode; credentials
are never changed by scripts. Select the phone and repeat the tiny-model UI tests
and manual lifecycle/memory checks. Record actual phone/OS, memory footprint,
load/unload release, power/thermal state and a bounded repeated run. Build-only
or simulator success does not qualify the phone.

Only after synthetic physical parity and measured headroom may a phone-specific
real-model admission budget be proposed. No large pretrained weights are bundled
or transferred here. Desktop memory/context qualification is not a phone budget.
See `docs/lessons/21-device-validation.md` for execution order, trace and exercise.
