# 21 — An iPhone host, with separate simulator and device gates

The phone host calls the same Rust inference engine through `CPUClient`.
Swift owns the screen and request admission; Rust owns weights, K/V, token
history, sampling and the forward pass. There is one worker and at most one
outstanding request. UI refreshes never copy the model or evaluate tensors.
The first app bundles the small synthetic model rather than pretrained weights.

Read `apple/iphone/README.md` for the Xcode project and exact follow-up commands.
On 2026-10-04, owner-approved Rust iOS targets, Xcode components and iOS27.0
arm64 runtime were installed. Release platform builds pass; the phone build is
unsigned. Both Simulator UI tests and 115 Rust numerical/bridge tests pass on
the actual M1Pro16GiB Mac running macOS26.6.2 and Xcode27.0. The simulated
iPhone14 runs iOS27.0/build24A434. Mac controller integration also passes.
No physical phone is connected; signing, phone memory/thermal measurements and
real-model qualification are not run. T21 and owner review remain pending.

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
7. Fatal failures clear displayed cache counters and tokens, disable further
   execution, and keep the screen Failed until recovery. Pause retains actual
   counters; failure does not imply that the last observed prefix is still valid.
   Scene transitions cannot replace Failed with an apparently resumable Paused.

The window's view observes scene phase and the controller publishes activity,
so an activity change refreshes button admission. Reading an unobserved flag
can leave Load disabled despite an active scene. Counter labels use verbatim
text so interpolation visibly preserves both numerical values. The Debug-only
memory-warning button posts the public UIKit notification; passing its listener
test is separate from actual OS pressure or physical memory release.

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
pretend that it was evaluated or advance counters. This trace passed in both the Mac
controller and Simulator UI tests; it remains expected for physical tests.

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

Exercise: a token request captures epoch 12. Unload advances it to 13, and a
new Load advances it to 14. The response from epoch 12 arrives afterward.
Should it change the current screen? Identify the guard that decides this and
explain why matching token IDs would not be enough. Do not implement the answer.
