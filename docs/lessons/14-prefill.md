# 14 — Reusing weights across prompt rows

Decode projects one row X[1,I]. Prompt projection can compute several X[T,I]
rows against the same W[O,I]. Independent row calls read each weight row again
for every token. The experiment changes loop order: token chunk, output row,
input group, token within chunk. One weight group is reused across that chunk.
The output remains row-major Y[T,O]; no hidden transpose occurs.

The rejected prototype first validates chunk1..8, all checked shape products and
exact slice lengths. Each output row gets eight possible four-lane accumulator
rows on the stack. For an active chunk it processes full input groups, updating
one accumulator per token. It reduces each row in the same order as T13, handles
the scalar input tail, then writes Y[(chunk_start+t)*O+o]. Only the active token
rows are read. A final short token chunk needs no padding.

**Trace:** For T=8/chunk2, chunk starts are0,2,4,6. At start2/output row3,
partial row0 projects X[2,:] and row1 projects X[3,:] against W[3,:]. Results go
to Y[2*O+3] and Y[3*O+3]. Weight-loop visits are four across the eight-token
input rather than eight, but this count is not measured DRAM traffic.

Measurements showed all chunk candidates slower than independent four-lane row
calls. The prototype executes extra loops and accesses stack accumulators, which
appears to cost more than the reuse saved. We reject production adoption and
preserve the development experiment for learning. Mechanical sympathy includes
instruction dependencies, compiler output and local work, as well as fewer
weight reads. Actual end-to-end prefill still uses T09/T12 serial cache steps;
this experiment has no cache, mask, token sampling or full-model speedup claim.
See `docs/experiments/14-prefill.md` for results, commands and limitations.

**Exercise:** For T=10 and chunk4, write the chunk starts/active lengths and the
row-major output offsets for output channel2 in the final chunk. How many logical
complete weight passes occur? Which additional contracts would a full decoder
prefill need beyond this projection experiment?
