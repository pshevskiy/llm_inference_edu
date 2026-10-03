//! Narrow C ABI. The numerical crate continues to forbid unsafe code.
//! Opaque integer IDs are never reused; foreign callers cannot dereference a
//! model/session. Commands copy bounded inputs and execute on one Rust worker.
#![deny(unsafe_op_in_unsafe_fn)]

mod worker;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};
use worker::{Command, Prompt, Source};

pub mod code {
    pub const OK: u32 = 0;
    pub const PENDING: u32 = 1;
    pub const INVALID_INPUT: u32 = 2;
    pub const INVALID_HANDLE: u32 = 3;
    pub const BUSY: u32 = 4;
    pub const CANCELLED: u32 = 5;
    pub const MODEL_FAILED: u32 = 6;
    pub const EXECUTION_FAILED: u32 = 7;
    pub const PANIC: u32 = 8;
    pub const CLOSED: u32 = 9;
}
pub mod kind {
    pub const READY: u32 = 1;
    pub const STARTED: u32 = 2;
    pub const TOKEN: u32 = 3;
    pub const STOPPED: u32 = 4;
    pub const CANCELLED: u32 = 5;
    pub const RESET: u32 = 6;
    pub const ERROR: u32 = 7;
}
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub kind: u32,
    pub code: u32,
    /// UINT32_MAX when this event emits no token.
    pub token: u32,
    pub processed: u32,
    pub history: u32,
}
struct Engine {
    commands: SyncSender<Command>,
    events: Mutex<Receiver<Event>>,
    cancel: Arc<AtomicBool>,
    fatal: Arc<AtomicBool>,
    busy: AtomicBool,
}
static REGISTRY: OnceLock<Mutex<HashMap<u64, Arc<Engine>>>> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static WORKER_ACTIVE: AtomicBool = AtomicBool::new(false);
fn registry() -> &'static Mutex<HashMap<u64, Arc<Engine>>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}
fn engine(handle: u64) -> Result<Arc<Engine>, u32> {
    registry()
        .lock()
        .map_err(|_| code::PANIC)?
        .get(&handle)
        .cloned()
        .ok_or(code::INVALID_HANDLE)
}
fn boundary(action: impl FnOnce() -> Result<(), u32>) -> u32 {
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(())) => code::OK,
        Ok(Err(error)) => error,
        Err(_) => code::PANIC,
    }
}
struct Reservation;
impl Drop for Reservation {
    fn drop(&mut self) {
        WORKER_ACTIVE.store(false, Ordering::Release);
    }
}
fn open(source: Source) -> Result<u64, u32> {
    // Serialize admission with registry close/open. A closing worker retains
    // its reservation until weights/session have actually dropped.
    let mut handles = registry().lock().map_err(|_| code::PANIC)?;
    if !handles.is_empty()
        || WORKER_ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
    {
        return Err(code::BUSY);
    }
    let reservation = Reservation;
    let id = NEXT_ID
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_add(1))
        .map_err(|_| code::CLOSED)?;
    let (command_tx, command_rx) = sync_channel(1);
    let (event_tx, event_rx) = sync_channel(1);
    let cancel = Arc::new(AtomicBool::new(false));
    let fatal = Arc::new(AtomicBool::new(false));
    let result = Arc::new(Engine {
        commands: command_tx,
        events: Mutex::new(event_rx),
        cancel: cancel.clone(),
        fatal: fatal.clone(),
        busy: AtomicBool::new(true),
    });
    std::thread::Builder::new()
        .name("llm-cpu".into())
        .spawn(move || {
            let _reservation = reservation;
            let panic_events = event_tx.clone();
            if catch_unwind(AssertUnwindSafe(|| {
                worker::run(source, command_rx, event_tx, cancel, fatal.clone())
            }))
            .is_err()
            {
                fatal.store(true, Ordering::Release);
                let _ = panic_events.send(Event {
                    kind: kind::ERROR,
                    code: code::PANIC,
                    token: u32::MAX,
                    processed: 0,
                    history: 0,
                });
            }
            // Worker-local model/session have dropped before reservation release.
        })
        .map_err(|_| code::EXECUTION_FAILED)?;
    handles.insert(id, result);
    Ok(id)
}
fn submit(handle: u64, command: Command) -> Result<(), u32> {
    let e = engine(handle)?;
    if e.fatal.load(Ordering::Acquire) {
        return Err(code::EXECUTION_FAILED);
    }
    if e.busy
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(code::BUSY);
    }
    if e.commands.try_send(command).is_err() {
        e.busy.store(false, Ordering::Release);
        return Err(code::CLOSED);
    }
    Ok(())
}
fn aligned<T>(pointer: *const T) -> bool {
    !pointer.is_null() && (pointer as usize).is_multiple_of(align_of::<T>())
}
/// ABI version is independent of the model and does not allocate.
#[unsafe(no_mangle)]
pub extern "C" fn aie_abi_version() -> u32 {
    1
}

/// Copy a tiny JSON (kind0, <=8MiB) or UTF-8 Qwen directory (kind1, <=4096B).
/// Returning OK starts asynchronous loading; poll its Ready/Error event.
/// On immediate errors the handle output remains unchanged.
/// # Safety
/// `bytes` must be readable for `length` bytes for this call; `out_handle` must
/// be aligned, writable for one u64 and disjoint from input. No pointer escapes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aie_open(
    model_kind: u32,
    bytes: *const u8,
    length: usize,
    out_handle: *mut u64,
) -> u32 {
    boundary(|| {
        let limit = match model_kind {
            0 => 8 * 1024 * 1024,
            1 => 4096,
            _ => return Err(code::INVALID_INPUT),
        };
        if !aligned(out_handle) || bytes.is_null() || length == 0 || length > limit {
            return Err(code::INVALID_INPUT);
        }
        // SAFETY: bounded nonempty readable allocation required by caller;
        // copy ends the borrow before any foreign output is written.
        let owned = unsafe { std::slice::from_raw_parts(bytes, length) }.to_vec();
        let source = if model_kind == 0 {
            Source::Tiny(owned)
        } else {
            let path = String::from_utf8(owned).map_err(|_| code::INVALID_INPUT)?;
            if path.contains('\0') {
                return Err(code::INVALID_INPUT);
            }
            Source::Qwen(path.into())
        };
        let handle = open(source)?;
        // SAFETY: caller grants aligned writable u64, after complete admission.
        unsafe {
            out_handle.write(handle);
        }
        Ok(())
    })
}
/// Submit one bounded pretokenized request. UTF-8 requests use aie_begin_text.
/// # Safety
/// `ids` must be aligned/readable for `count` u32 values throughout this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aie_begin_ids(
    handle: u64,
    ids: *const u32,
    count: usize,
    max_new: u32,
) -> u32 {
    boundary(|| {
        if !aligned(ids) || count == 0 || count > 32 || max_new == 0 || max_new >= 32 {
            return Err(code::INVALID_INPUT);
        }
        // SAFETY: bounded readable aligned allocation is the caller contract.
        let owned = unsafe { std::slice::from_raw_parts(ids, count) }
            .iter()
            .map(|&v| v as usize)
            .collect();
        submit(handle, Command::Begin(Prompt::Ids(owned), max_new as usize))
    })
}
/// Plain UTF-8 prompt; Rust verifies tokenization assets and encodes on worker.
/// Chat/tool rendering is outside this initial native boundary.
/// # Safety
/// `bytes` must be readable for `length` bytes for this call; no pointer escapes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aie_begin_text(
    handle: u64,
    bytes: *const u8,
    length: usize,
    max_new: u32,
) -> u32 {
    boundary(|| {
        if bytes.is_null() || length == 0 || length > 65536 || max_new == 0 || max_new >= 32 {
            return Err(code::INVALID_INPUT);
        }
        // SAFETY: bounded readable allocation required by caller; copied now.
        let text = std::str::from_utf8(unsafe { std::slice::from_raw_parts(bytes, length) })
            .map_err(|_| code::INVALID_INPUT)?
            .to_owned();
        submit(handle, Command::Begin(Prompt::Text(text), max_new as usize))
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn aie_next(handle: u64) -> u32 {
    boundary(|| submit(handle, Command::Next))
}
#[unsafe(no_mangle)]
pub extern "C" fn aie_reset(handle: u64) -> u32 {
    boundary(|| submit(handle, Command::Reset))
}
/// Immediate cooperative cancellation; enabled0 explicitly resumes. Cancellation
/// can be set while loading/stepping, without waiting behind the command queue.
#[unsafe(no_mangle)]
pub extern "C" fn aie_set_cancel(handle: u64, enabled: u32) -> u32 {
    boundary(|| {
        if enabled > 1 {
            return Err(code::INVALID_INPUT);
        }
        engine(handle)?
            .cancel
            .store(enabled == 1, Ordering::Release);
        Ok(())
    })
}
/// Poll without blocking. OK writes exactly one event and frees admission;
/// PENDING/errors leave output unchanged. Event buffer remains caller-owned.
/// # Safety
/// `out_event` must be aligned/writable for one Event for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aie_poll(handle: u64, out_event: *mut Event) -> u32 {
    boundary(|| {
        if !aligned(out_event) {
            return Err(code::INVALID_INPUT);
        }
        let e = engine(handle)?;
        let event = match e.events.lock().map_err(|_| code::PANIC)?.try_recv() {
            Ok(value) => value,
            Err(TryRecvError::Empty) => return Err(code::PENDING),
            Err(TryRecvError::Disconnected) => return Err(code::CLOSED),
        };
        // SAFETY: caller grants one aligned writable Event; only writes after
        // complete pointer/handle/event admission, and no foreign borrow retained.
        unsafe {
            out_event.write(event);
        }
        e.busy.store(false, Ordering::Release);
        Ok(())
    })
}
/// Nonblocking close: invalidate ID immediately, cancel and drop the channels.
/// Worker releases resources after an in-flight load/token boundary. A new model
/// is BUSY until the closing worker's model/session have actually dropped.
#[unsafe(no_mangle)]
pub extern "C" fn aie_close(handle: u64) -> u32 {
    boundary(|| {
        let e = registry()
            .lock()
            .map_err(|_| code::PANIC)?
            .remove(&handle)
            .ok_or(code::INVALID_HANDLE)?;
        e.cancel.store(true, Ordering::Release);
        Ok(())
    })
}

#[cfg(test)]
mod tests;
