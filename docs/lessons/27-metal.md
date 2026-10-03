# 27 — One residual addition on Metal

The owner authorized T27. We start with an operation already understood on CPU:
`y[i] = x[i] + residual[i]`. The scalar engine stays available and unchanged.
This is one Metal operation, not a GPU decoder or a claim of faster inference.
The educational focus remains the owner's understanding of LLM inference.

## Execution order

1. `ResidualAdd::new(capacity)` checks positive capacity, checked FP32 byte
   arithmetic, the experiment's maximum1,048,576elements, and the device's buffer
   limit. Three shared buffers occupy at most12MiB of logical payload.
2. The public Rust binding creates the device, queue, compiled shader and pipeline
   once. Fast math is disabled. Threadgroup width comes from pipeline limits;
   this M1 Pro reports32. Nil allocations are checked before making Rust owners.
3. `run` validates exact slice lengths, capacity and finite inputs. It copies the
   two Rust inputs into private Metal-owned shared buffers. Unified memory avoids
   a separate discrete-device transfer, but these are still real CPU copies.
4. The encoder binds input/output buffers and a copied32-bit length. It dispatches
   `ceil(n/group_width)` groups. Adjacent GPU threads read adjacent FP32 values:
   the memory layout matches neighboring work rather than strided row reads.
5. The host commits the command and waits. Shared storage alone does not make
   concurrent CPU/GPU access safe. Only successful completion permits output
   inspection. A command failure invalidates this executor.
6. The host checks every output is finite, then copies into caller-owned output.
   All errors preserve that output. An autorelease pool releases temporary
   Objective-C command objects after every synchronous call.

Mapped-pointer operations and Objective-C selectors are isolated in
`apple/metal_slice/src/macos.rs`, with local alignment, length, ownership,
signature and synchronization invariants. No mapped Rust reference survives a
dispatch. `&mut self` prevents simultaneous use through this safe interface;
caller slices are never lent to the GPU. Root `src/lib.rs` still forbids unsafe.

## Trace

Inputs `[1,-2,0,4]` and `[2,3,-5,-4]` produce `[3,1,-5,0]`.
With32threads/group, one group runs. Thread3 reads byte offset12 from each input,
computes `4 + (-4)`, and writes0 to output byte offset12. Threads4–31 see
`index >= n` and return without touching storage. The CPU reads those four
outputs only after the command completes.

## Numerical limits

The finite-value contract rejects NaN/infinite inputs, and finite-input overflow
rejects before copying output. Normal-value fixture results are exact. The frozen
qualification gate is `atol=1e-6, rtol=1e-6`; no model/logit gate changes.
Signed-zero and cancellation behavior are checked separately by their bits.
Despite disabled fast math, this GPU flushes subnormal inputs/results. For
example, two smallest positive FP32 values sum to0 on GPU and bits2 on CPU.
`MIN_POSITIVE + (-MIN_POSITIVE/2)` remains `MIN_POSITIVE` on this GPU, because
the subnormal input is flushed. These tiny differences pass the absolute gate;
that does not establish complete IEEE identity. No decoder adopts this operation.

## What the measurement teaches

The public-call time includes validation, both input copies, command creation,
submission/wait, output validation and output copy. Command-buffer GPU timestamps
cover a smaller interval. Scalar timing includes its public validation and loop.
Pipeline setup, input construction and comparison are outside repeated timings.
At1,048,576elements the GPU command itself is short, but the full public call is
slower than scalar Rust. Shared memory and a parallel shader are insufficient to
establish an application speedup. Keeping intermediate tensors on GPU or fusing
work would require separate code, lifetime contracts and new evidence.

Run the separately opt-in, bounded qualification from the repository root:

```sh
cargo build --offline --release -p llm_metal_slice --example residual
/usr/bin/time -l target/release/examples/residual --run docs/evidence/my-metal-run
```

Use a fresh output directory. Ordinary tests need no GPU execution. Missing GPU
or shader compilation yields failed completion evidence; it never falls back to
CPU and claims Metal passed. See experiment27 for this actual Mac's results.

Exercise: for97elements and32threads/group, how many groups are dispatched, and
which thread indices must return without writing? Explain why the length guard
is still necessary even when the allocated buffer is larger than97elements.

References: Apple's [shared-resource synchronization](https://developer.apple.com/documentation/metal/mtlstoragemode/shared),
[compute dispatch](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder),
[completion wait](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted()),
and the pinned [Rust binding source](https://github.com/gfx-rs/metal-rs/tree/v0.33.0).
