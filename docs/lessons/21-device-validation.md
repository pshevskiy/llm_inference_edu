# 21 — An iPhone host, with separate simulator and device gates

The phone host calls the same Rust inference engine through `CPUClient`.
Swift owns the screen and request admission; Rust owns weights, K/V, token
history, sampling and the forward pass. There is one worker and at most one
outstanding request. UI refreshes never copy the model or evaluate tensors.
The first app bundles the small synthetic model rather than pretrained weights.

Read `apple/iphone/README.md` for the Xcode project and exact follow-up commands.
On 2026-10-04, Swift 6 source typechecking passed against the simulator and
physical iOS SDKs. Controller integration checks passed against the real Rust
bridge **on the Mac**. Xcode project loading failed because CoreSimulator is
missing, and neither Rust iOS target is installed. No simulator executable was
linked or run; no phone was signed, installed or tested. Owner review is pending.

## The central controller in execution order

1. `PhoneSession.load` admits work only while active and without an existing
   client. It captures an epoch and reads at most 8 MiB + 1 bytes off the main
   thread. Oversize, empty and missing files fail explicitly.
2. After that read, an unchanged epoch permits `CPUClient` to open the tiny Rust
   session. Loading and inference run on its worker. READY arrives on the main
   queue; it enables actions while preserving any background/pause decision.
3. `prepare` submits `[1,5,7]` with a four-token output limit. While busy, duplicate
   actions are rejected. Responses carry the actual evaluated-prefix/history
   counters; Swift does not guess cache progress.
4. `step` asks Rust for one token. At most four displayed token IDs are retained
   in Swift. Rust retains the complete, exact generation state.
5. `pause` sets cancellation but leaves an in-flight response outstanding.
   A token boundary may already have finished. Returning from background never
   resumes automatically; explicit resume clears cancellation. Finished stays
   finished even after pause/resume. Reset clears generation and restores pause
   cancellation if backgrounding raced the reset.
6. `unload` increments the epoch before closing the handle. Late callbacks or a
   late fixture read are discarded. Model release is asynchronous; a new open
   can report busy until the old worker releases its reservation. Memory warnings
   call this same close-and-clear path. Cleared counters do not claim a valid K/V
   prefix or a resumable transcript.

This follows mechanical sympathy by keeping numerics off the screen thread,
reusing the existing worker/cache, preventing request backlogs, and avoiding
parallel model copies. It is a resource design, not measured phone efficiency.
A simulator runs on Mac hardware; physical memory, bandwidth and thermal limits
still need measurement. No numerical kernel optimization is claimed here.

## One concrete trace

The independently checked tiny fixture emits `[41,53,41,50]`:

| Action | Evaluated prefix | History | Displayed tokens |
|---|---:|---:|---|
| Prepare `[1,5,7]` | 3 | 3 | empty |
| Emit `41` | 3 | 4 | 41 |
| Background, then explicitly resume | 3 | 4 | 41 |
| Next step evaluates `41`, then emits `53` | 4 | 5 | 41, 53 |
| Fourth emission | 6 | 7 | 41, 53, 41, 50 |
| Observe output-limit stop | 6 | 7 | unchanged |
| Reset | 0 | 0 | empty |

The last emitted token stays pending at the output limit. A pause must not
pretend that it was evaluated or advance counters. This trace passed on the Mac
controller; it remains an expected trace for simulator and physical tests.

## Why two arm64 libraries?

`aarch64-apple-ios-sim` and `aarch64-apple-ios` share an instruction architecture
but have different Mach-O platform identities. Rust and every native dependency
must be cross-built for the chosen platform. The Xcode build stores archives in
separate platform directories; it never links the Mac library into the phone.
The current slice accepts arm64 only and fails clearly on unsupported requests.
Scripts do not install targets or change accounts/signing settings.

## Physical qualification remains open

First pass simulator build/UI/lifecycle checks; then build/sign using the owner's
team for a reachable, trusted Developer Mode iPhone 14. Run the same synthetic
numerical trace, cancellation, load/unload, background/foreground and memory
checks on that actual phone. Measure release rather than just removing a handle.
Keep signed build, simulator and physical results separately labelled.

Only then assess actual headroom, logical weights/cache/workspace and measured
footprint for a pinned real model. Desktop C32 or 16 GB RAM does not define a
phone budget. If it does not fit, report blocked rather than changing model,
precision or history silently. Record phone/OS, thermal/power state, memory
signals and bounded repeated-run counters; label unavailable measurements.

Exercise: backgrounding occurs after emitting `41` at prefix/history `3/4`, then
a memory warning closes the session. Which counters should the screen show,
and can tapping Load resume using the old K/V? Explain the difference between
a displayed token list and valid cache storage. Do not implement the answer yet.
