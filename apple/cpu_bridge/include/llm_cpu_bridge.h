#ifndef LLM_CPU_BRIDGE_H
#define LLM_CPU_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI2. IDs never reused. Only one live/loading/closing model worker.
 * Caller owns all memory. Inputs must be readable for their explicit length;
 * typed pointers must be aligned; outputs writable and disjoint from inputs.
 * Rust copies inputs during the call and retains no foreign pointers.
 * Invalid pointers cannot be made safe by an FFI error code or panic catcher.
 */
enum {
    AIE_OK=0, AIE_PENDING=1, AIE_INVALID_INPUT=2, AIE_INVALID_HANDLE=3,
    AIE_BUSY=4, AIE_CANCELLED=5, AIE_MODEL_FAILED=6, AIE_EXECUTION_FAILED=7,
    AIE_PANIC=8, AIE_CLOSED=9
};
enum {
    AIE_READY=1, AIE_STARTED=2, AIE_TOKEN=3, AIE_STOPPED=4,
    AIE_CANCEL_EVENT=5, AIE_RESET_EVENT=6, AIE_ERROR=7
};
typedef struct {
    uint32_t kind, code, token, processed, history;
} AieEvent;
/* token is UINT32_MAX for events which emit none. Successful token event:
 * history == processed +1, unless cancellation preserves a prompt prefix.
 */
uint32_t aie_abi_version(void);
/* kind0: tiny JSON <=8MiB; kind1: verified Qwen directory UTF8 <=4096B.
 * Open returns promptly; poll its Ready/Error event before submitting work.
 * On immediate error out_handle remains unchanged. Allocation OOM/process
 * signals are not recoverable panics. Numerical crate forbids unsafe code.
 */
uint32_t aie_open(uint32_t kind, const uint8_t *bytes, size_t length, uint64_t *out_handle);
/* At most one outstanding command/event. Begin requires empty/reset history.
 * Whole prompt/output budget validated on worker before numerical writes.
 * Plain UTF8 text is tokenized in Rust; chat/tool rendering is not added here.
 */
uint32_t aie_begin_ids(uint64_t handle, const uint32_t *ids, size_t count, uint32_t max_new);
uint32_t aie_begin_text(uint64_t handle, const uint8_t *bytes, size_t length, uint32_t max_new);
uint32_t aie_next(uint64_t handle);
uint32_t aie_reset(uint64_t handle);
/* enabled1 cancels at token boundaries; enabled0 explicitly resumes.
 * In-flight load/step completes; no partially valid KV exposed. */
uint32_t aie_set_cancel(uint64_t handle, uint32_t enabled);
/* Nonblocking. OK writes one event and opens command admission;
 * PENDING/error leaves out_event unchanged. */
uint32_t aie_poll(uint64_t handle, AieEvent *out_event);
/* ABI2: atomically poll metadata plus UTF8 chunk. Output allocations must be
 * mutually disjoint, metadata aligned; text capacity must be exactly4096.
 * PENDING/errors leave all outputs untouched and consume no event. Only first
 * out_length bytes are written; output is not NUL terminated. Tiny emits IDs.
 * Plain real prompts+output are capped at8positions until larger qualification.
 */
uint32_t aie_poll_text(uint64_t handle, AieEvent *out_event,
    uint8_t *text, size_t capacity, size_t *out_length);
/* Nonblocking invalidation; actual model release is asynchronous. New opens
 * return BUSY until the closing worker has released its model/session. */
uint32_t aie_close(uint64_t handle);
#ifdef __cplusplus
}
#endif
#endif
