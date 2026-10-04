# Inference Lab for iPhone — T21 development slice

A small SwiftUI host for the existing Rust CPU engine. It bundles only the
435,741-byte synthetic model, prepares `[1,5,7]`, and requests four greedy tokens
one at a time. It exposes evaluated-prefix and history counts, pause/resume,
reset and unload. There is no pretrained-model installer or conversational
assistant in this slice. No new Rust dependencies or parallel inference engine.

**Current validation:** Swift source typechecks against the simulator and
physical iOS SDKs; the real-bridge controller test passes on this M1 Mac.
Simulator app linking/UI execution and physical-device validation are **not
run**: Rust iOS targets are absent and Xcode's CoreSimulator framework is missing.
Project/plist syntax checks pass, but Xcode project loading itself is blocked
by the missing framework. This is source/build configuration, not a certified
runnable phone app. T21 remains unchecked; owner review pending.

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
Run the checked-in UI test (replace `SIMULATOR_UUID`):

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild \
  -project apple/iphone/InferencePhone.xcodeproj -scheme InferencePhone \
  -configuration Debug -destination 'platform=iOS Simulator,id=SIMULATOR_UUID' \
  -derivedDataPath target/iphone-xcode \
  -resultBundlePath docs/evidence/t21-simulator.xcresult \
  CODE_SIGNING_ALLOWED=NO test
```

The test checks `[41,53,41,50]`, exact counters, background/foreground pause,
explicit resume, finished-state preservation, reset and unload. It has not run.
Once running, manually test Simulator → Debug → Simulate Memory Warning during
load and token requests. Confirm state becomes Unloaded, old callbacks do not
restore it, and reload starts an empty session. Record screenshots and actual
runtime/device identity. Actual OS memory pressure is a separate physical test.

## Mac controller check

```sh
/bin/sh apple/iphone/check-controller.sh
```

This builds the Mac Rust bridge and controller with Swift 6/warnings as errors.
It tests independent expected tokens/counters, duplicate-request rejection,
pause during load and requests, background/foreground during prompt/reset,
memory-warning handler closure, stale callbacks, reload and missing-resource
failure. It does **not** simulate iOS lifecycle delivery or phone memory pressure.

## Physical iPhone after simulator validation

Connect/trust the iPhone 14, enable Developer Mode, and check Xcode Window →
Devices and Simulators. The owner selects their signing team in Xcode; credentials
are never changed by scripts. Select the phone and repeat the tiny-model UI tests
and manual lifecycle/memory checks. Record actual phone/OS, memory footprint,
load/unload release, power/thermal state and a bounded repeated run. Build-only
or simulator success does not qualify the phone.

Only after synthetic physical parity and measured headroom may a phone-specific
real-model admission budget be proposed. No large pretrained weights are bundled
or transferred here. Desktop memory/context qualification is not a phone budget.
See `docs/lessons/21-device-validation.md` for execution order, trace and exercise.
