# T28 — CPU context and actual calculator continuation

All six stages through 2,048 positions passed their numerical, greedy-token,
sampled-continuation and measured resource gates. Owner review remains pending.

The M1 Pro (MacBookPro18,3, 16 GiB, macOS 26.6.2/build25G83) runs the pinned
Qwen revision `7ae557604adf67be50417f59c2c2f167def9a775`, BF16 source weights
expanded to FP32. Rust 1.90.0/Homebrew LLVM21.1.1, scalar CPU forward pass with
the previously accepted safe four-lane vocabulary projection. No GPU, quantized
KV, context extension, model substitution or tolerance relaxation.

Independent reference: torch2.6.0/Transformers4.46.3/tokenizers0.20.3/
NumPy2.3.2/safetensors0.6.2, CPU FP32 eager, one torch thread. Every vocabulary
value at each forced position, finite outputs, exact greedy IDs. Frozen gate:
`abs(actual−reference) ≤ 0.001 + 0.0001×abs(reference)`. Selected full-prefix
calibrations retain only the last vocabulary row and explicitly account for
quadratic reference attention storage. Cached production Rust uses linear scratch.

| Allocated capacity | Source snapshot | Prompt in timed sample | Prefill s | Decode tokens/s | Continuation s | Logical KV MiB | Peak process footprint GiB | Worst normalized logit budget |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | A | 20 | 6.626 | 3.024 | 1.654 | 0.75 | unavailable | 0.758759 |
| 128 | A | 116 | 38.559 | 3.000 | 1.665 | 3 | 2.773 | 0.758759 |
| 256 | A | 244 | 80.555 | 3.007 | 1.660 | 6 | 2.776 | 0.758759 |
| 512 | B | 500 | 166.315 | 2.948 | 1.695 | 12 | 2.782 | 0.758759 |
| 1024 | B | 1012 | 343.524 | 2.833 | 1.766 | 24 | 2.794 | 0.758759 |
| 2048 | B | 2036 | 721.824 | 2.627 | 1.903 | 48 | 2.834 | 0.758759 |

Source snapshots are distinct; the table does not imply a single identical build:

- A source SHA256: `5894205269f21c7e9c2dd35a6c7c16d2bcfb0cd2b80b35412a7af9000e5e4561`;
  executable: `95381f76f3d80c568dbcbd76655b26fbd2d6f6ea1995a41f493efc8d74c53b0f`.
- B (current) source SHA256: `48aafed5ff1592e5385d16f356fac5f37ec36b19d0ebde24c01258d8755f7733`;
  executable: `42818e7a9a402daea84a287c7ec0141a2310e40e563c0b1f93d5a912e548ec70`.

Between these groups, CLI integration was finalized and the qualifier's final
result changed from hashing the executable path again at completion to copying
the identity captured at startup. Earlier source bytes were not separately
archived, so a complete byte-for-byte intervening diff is not claimed. The
earlier figures remain evidence for A; B independently passes all three larger
stages, the actual calculator probes, and portable differing-stride tests.

These are **single bounded warm request samples**, not medians/p95s, sustained
thermal results or a backend comparison. Model loading/cache construction and
numerical comparisons are outside request timing. Three independent expected
rows follow the actual sampled request after it diverges from teacher forcing.
Continuation evaluates one pending emitted token followed by four fixture IDs,
retaining the existing prefix; its next emission remains pending. The first32
sandboxed engine run passed, but its measurement wrapper failed with `sysctl
kern.clockrate: Operation not permitted`; its physical figures are unavailable.
Later stages use permitted actual process observation. Maximum RSS and macOS
peak process footprint are distinct fields; table shows the latter. Energy is
unavailable. See raw logs, identity and position records under ignored evidence.

At capacity2,048, maximum RSS is 2,884,534,272 bytes and peak process footprint
is 3,043,199,688 bytes. After two emissions, evaluated/history lengths are
2,037/2,038. Appending four IDs evaluates the pending token and suffix once,
reaching2,042/2,042; the next emission leaves2,042/2,043. The three independent
sampled-request numerical budgets are0.061203/0.052564/0.050779, all greedy IDs
exact. Full capacity2,048 numerical pass plus request took1,472.06seconds; that
whole-process duration includes comparison work and is not a request latency.

Actual calculator probe uses the previously validated no-tools plain-message
chat branch. A typed calculator call is validated before execution; the numeric
envelope is appended as a user data message. Explicit EOS is already in history
and is evaluated once before that suffix. Three fixed probes pass independently:

| Question | Requested operation/arguments | Final answer | Worst normalized numerical budget |
|---|---|---|---:|
| 2 plus 3 | add(2,3) | 5 | 0.231009 |
| 7 minus 2 | subtract(7,2) | 5 | 0.335222 |
| 6 divided by 2 | divide(6,2) | 3 | 0.242342 |

For addition:86promptIDs, typed request ends at evaluated109/history110; pending
EOS plus20result/framingIDs reaches130/130; final JSON+EOS ends at139/140.
Every emitted full vocabulary row/token agrees with the independent greedy
reference. Rust actual-agent peak process footprint3,034,467,064bytes. Numerical
agreement and schema/argument/answer quality are reported separately. **Three
fixed probes establish no general agent reliability, teaching quality or long
retrieval capability.** The long-stage timed suffix is a fixture, not a claimed
model-selected calculator call at every tested position.

Identity is recorded before execution and completion/failure separately. The
reference streams full position logits to disk (file size is not RAM usage).
Binary identity is captured at startup. Fresh result directories preserve earlier
runs. The first 2,048-position reference completed with shared-memory diagnostics
and is preserved. Its identical-configuration repeat outside the sandbox
completed without those diagnostics; every streamed logit and request oracle is
identical. The final Rust stage uses that repeated reference; raw first-run
evidence is retained.

CLI explicit context512 check returns5+EOS, evaluated25/history26. Native app
budget remains8 withC32allocation; its previous compiled artifact is unchanged.
Phone validation remains blocked: no installed iOS Rust standard library, and
Xcode's devicectl shim cannot find CoreDevice (component request failed).
No signing change/toolchain installation/device execution was performed.

Reproduction: `benchmarks/README.md` and lesson28. Current source/binary identity,
host, raw samples, physical measurements, reference hashes and per-position
numbers are in `docs/evidence/t28-*`, excluded from Git. Owner review pending;
Astra final review approved, with no remaining actionable findings. Astra audited
the saved position streams, gates, continuation counters and source snapshots;
it ran no additional model or hardware workloads.
