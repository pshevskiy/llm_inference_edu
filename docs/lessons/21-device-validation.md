# 21 — Physical-device qualification is still blocked

A Mac static library is not an iPhone library. Rust code and every native
component (including the tokenizer's C dependency) must be built for
`aarch64-apple-ios`; a simulator uses a different target. The Swift application
must then be built and signed with the owner's development team and provisioned
for a reachable physical iPhone 14. None of these phone results are inferred
from the successful Mac boundary checks.

Readiness observed on 2026-10-03:

- Installed Xcode 27.0, iPhoneOS SDK27.0; global developer path now points at
  Xcode. This task did not change that selection.
- Installed Rust targets: `aarch64-apple-darwin`, `wasm32-wasip2`,
  `x86_64-unknown-linux-musl`. No iOS target is installed.
- Installed Xcode `devicectl list devices` produced no output within a15second
  deadline; the owned child group was terminated. Device availability is
  unknown. Earlier unbounded readiness commands were interrupted too.
- Account, signing identity and provisioning were not inspected or changed.
  No physical phone app was built, installed, signed or executed.

Follow-up readiness checks (read-only):

```sh
rustup target list --installed
xcrun --sdk iphoneos --show-sdk-version
```

In Xcode's **Window → Devices and Simulators**, check that the owner's physical
phone is connected, trusted, in Developer Mode and available. This manual check
is needed because CLI enumeration currently hangs. Installing the Rust iOS
target requires an explicit owner decision under the original no-automatic-
toolchain-installation instruction. After it exists, the concrete bridge build
check is:

```sh
cargo build --locked --release -p llm_cpu_bridge --target aarch64-apple-ios
```

That command is not yet run and may expose tokenizer C cross-build requirements.
Do not reuse the Mac archive or count a simulator result as physical validation.
A signed device host still needs to be built with the owner's configured team;
there is no claimed runnable iOS target in this repository yet.

When prerequisites are available, begin with the tiny fixture. Run the exact
T20 token/counter trace and cancellation/resume/reset, and unload/reload; measure
actual memory release rather than just removing a handle. Inject backgrounding
at load, prefill and token boundaries; stop/suspend new requests predictably,
then verify valid-prefix reuse or explicit reset. Test low-memory warnings with
small synthetic data before allocating pretrained weights.

Only then propose a phone-specific real-model admission budget using actual
available headroom, measured footprint and logical weights/cache/workspace.
Desktop C32 and16GB headroom do not define an iPhone budget. Load the already
pinned checkpoint only if the budget admits it; otherwise mark real-model
validation blocked rather than silently switching checkpoint or dtype. Retain
short exact token IDs and teacher-forced numerical checks, offline behavior,
load/unload/cancellation/suspension evidence and an explicitly bounded repeated
run. Capture physical device/OS, power, thermal, memory-pressure signals and
raw counters; write unavailable where the instrument cannot measure a field.

Trace to verify on a future phone: prompt `[1,5,7]` evaluates3positions;
emitting41 produces history4 with valid prefix3. Background suspension must not
claim prefix4 before41 was evaluated. This is a proposed check, not a device
result.

Exercise: at that3/4 state, choose which counters and volatile resources must be
saved or invalidated before a phone memory warning closes the worker. Do not
assume persistent transcript is the same thing as K/V.
