# 23 — Identity and atomic installation

A search result, checkpoint and execution artifact are three different things.
An MLX quantization is not the exact BF16 checkpoint used by our CPU engine.
`CPURecipe.compatibility` recognizes only the exact pinned Qwen repository and
revision. Other repositories remain discovery-only; a different revision of the
selected repository is unsupported, and missing revision is unknown.

The checked-in source inventory pins nine names, sizes and SHA256 digests. The
separate execution record pins the reviewed Mac static archive and native
parity executable. It declares FP32 CPU/C32 allocation and the narrow short
real-model validation, with ANE/iPhone unqualified. These are trusted application
inputs, never downloaded repository code/config/template instructions.

Runnable requires both a `VerifiedInstallation` and `VerifiedExecution` matching
the recipe. Source-only verification yields recipe-available. Execution
verification checks actual archive/executable bytes against their expected
hashes and executable permissions; copied metadata alone is insufficient.
These capabilities describe verification at that point in time, not immunity
to later filesystem changes. The Rust loader still checks source assets at load.
The artifact record is tied to the reviewed native Mac build (T24 supersedes the T20 baseline), not every future build.
A rebuilt executable/archive needs its own qualification and reviewed digest.

Execution order:

1. Validate the trusted pin, fixed asset list, simple top-level filenames,
   nonzero sizes and digests, using checked total/budget arithmetic.
2. Admit one install task. Disk I/O and hashing run detached from UI; cancellation
   remains responsive through the installer actor. Existing destination is
   verified and reused; a corrupt destination is reported and never overwritten.
3. Check destination-volume available capacity against twice source payload
   plus64MiB, conservatively allowing temporary/copy overhead. This is a disk
   snapshot, not an allocation guarantee or runtime-memory estimate. A separate
   URLSession temporary volume can still run out of space and fail safely.
4. Create a private UUID staging directory beside the eventual revision
   directory. Fetch each asset using its pinned resolve URL. Public anonymous
   URLSession downloads write to disk; redirects stay on HTTPS Hub/CDN domains.
   No gated authentication, cookies, credentials or repository code execution.
5. Cancel downloads on observed excess bytes or expected transfer length.
   Progress callbacks happen after writes, so this is not a strict byte ceiling
   on disk: one in-flight chunk may overshoot. Final exact size is mandatory.
   Move the OS temporary file during its guaranteed delegate-callback lifetime.
6. Hash each staged file in4MiB chunks, with cancellation checks. Reject size,
   hash, nonregular file or symlink mismatch. No full-weight Data buffer is kept.
7. Check cancellation again, then rename the complete directory to its final
   name on the same volume. This is the single publication point. Clean up only
   this task's staging directory on error/cancellation. Crashes can leave an
   ignored `.partial` directory; it is never treated as installed. Atomic
   visibility is intended; power-loss durability/fsync is not claimed.

Trace: download LICENSE into stage S and verify it. A later tokenizer asset
fails its digest. The task removes S, leaves the previous source/install intact,
and publishes nothing. A retry gets a new staging directory. Offline tests also
hold a partial file, cancel, verify cleanup, then successfully install a retry.

Build/run small network-free contracts (no ignored binaries/models needed):

```sh
xcrun swiftc -swift-version 6 -warnings-as-errors -O -parse-as-library \
  apple/HubSearch.swift apple/ModelInstallation.swift apple/InstallationSmoke.swift \
  -o /tmp/llm-install-check
/tmp/llm-install-check
```

Tests use tiny hand-built source bytes and mock execution bytes/digests, never
load them as Qwen, and do not claim numerical qualification. A separately opt-in
`--network-small` check fetches only the pinned11KB LICENSE, checks size/hash,
observed-overrun rejection and cancellation before starting transfer. It does
not test cancellation during a sustained large download or actual1GB network
installation; those are not run.

`--local` stages/reuses the already-downloaded source at
`models/installed-qwen/<revision>` and checks the recorded native workbench execution files.
It requires those exact ignored qualification artifacts on this tested Mac;
fresh clones/new builds get recipe-available until independently qualified.
The installed source was subsequently loaded by the checked native executable
and produced exact `[358,2776,264,48948]` with the T20 lifecycle counters.

Exercise: a discovered MLX4bit repository says its base model is our pinned Qwen.
List which identity and verification facts are still missing before it can be
called runnable. Do not add a conversion recipe automatically.
