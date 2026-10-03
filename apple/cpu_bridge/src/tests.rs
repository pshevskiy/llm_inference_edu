use super::*;
use std::time::{Duration, Instant};
static SERIAL: Mutex<()> = Mutex::new(());
const TINY: &[u8] = include_bytes!("../../../fixtures/tiny-model.json");
fn event() -> Event {
    Event {
        kind: 99,
        code: 99,
        token: 99,
        processed: 99,
        history: 99,
    }
}
fn wait(handle: u64) -> Event {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let mut out = event();
        // SAFETY: out is a unique live stack Event of the correct alignment.
        let status = unsafe { aie_poll(handle, &mut out) };
        if status == code::OK {
            return out;
        }
        assert_eq!(status, code::PENDING);
        assert_eq!(out, event());
        assert!(Instant::now() < deadline, "worker event timeout");
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn ready() -> u64 {
    let mut handle = 0;
    // SAFETY: fixture is readable, handle is a disjoint writable stack u64.
    assert_eq!(
        unsafe { aie_open(0, TINY.as_ptr(), TINY.len(), &mut handle) },
        code::OK
    );
    assert_eq!(wait(handle).kind, kind::READY);
    handle
}
fn close(handle: u64) {
    assert_eq!(aie_close(handle), code::OK);
    let deadline = Instant::now() + Duration::from_secs(5);
    while WORKER_ACTIVE.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}
#[test]
fn public_abi_layout_and_pre_admission_outputs_are_preserved() {
    let _serial = SERIAL.lock().unwrap();
    assert_eq!(aie_abi_version(), 2);
    assert_eq!(size_of::<Event>(), 20);
    assert_eq!(align_of::<Event>(), 4);
    let mut h = 99;
    // SAFETY: the live output is writable; invalid null/length inputs are
    // rejected before any attempted dereference.
    unsafe {
        assert_eq!(
            aie_open(0, std::ptr::null(), 1, &mut h),
            code::INVALID_INPUT
        );
        assert_eq!(
            aie_open(0, TINY.as_ptr(), usize::MAX, &mut h),
            code::INVALID_INPUT
        );
    }
    assert_eq!(h, 99);
    let mut out = event();
    // SAFETY: live output; invalid handle is a checked integer, never a pointer.
    assert_eq!(unsafe { aie_poll(0, &mut out) }, code::INVALID_HANDLE);
    assert_eq!(out, event());
    assert_eq!(aie_close(0), code::INVALID_HANDLE);
    assert_eq!(boundary(|| panic!("boundary test")), code::PANIC);
}
#[test]
fn synthetic_parity_bounded_delivery_cancel_resume_reset_and_stale_ids() {
    let _serial = SERIAL.lock().unwrap();
    let h = ready();
    let prompt = [1_u32, 5, 7];
    // SAFETY: prompt is live/aligned for all three elements, copied in this call.
    assert_eq!(
        unsafe { aie_begin_ids(h, prompt.as_ptr(), prompt.len(), 4) },
        code::OK
    );
    assert_eq!(aie_next(h), code::BUSY); // Cannot accumulate commands/events.
    let started = wait(h);
    assert_eq!(
        (started.kind, started.processed, started.history),
        (kind::STARTED, 3, 3)
    );
    for (index, expected) in [41, 53, 41, 50].into_iter().enumerate() {
        if index == 1 {
            assert_eq!(aie_set_cancel(h, 1), code::OK);
            assert_eq!(aie_next(h), code::OK);
            let cancelled = wait(h);
            assert_eq!(
                (cancelled.kind, cancelled.processed, cancelled.history),
                (kind::CANCELLED, 3, 4)
            );
            assert_eq!(aie_set_cancel(h, 0), code::OK);
        }
        assert_eq!(aie_next(h), code::OK);
        let token = wait(h);
        assert_eq!((token.kind, token.token), (kind::TOKEN, expected));
        assert_eq!(token.processed, 3 + index as u32);
        assert_eq!(token.history, 4 + index as u32);
    }
    assert_eq!(aie_next(h), code::OK);
    assert_eq!(wait(h).kind, kind::STOPPED);
    assert_eq!(aie_reset(h), code::OK);
    assert_eq!(wait(h).history, 0);
    let invalid = [64_u32];
    // SAFETY: invalid ID is in a valid readable one-element buffer.
    assert_eq!(
        unsafe { aie_begin_ids(h, invalid.as_ptr(), 1, 4) },
        code::OK
    );
    let rejected = wait(h);
    assert_eq!(
        (rejected.kind, rejected.history, rejected.processed),
        (kind::ERROR, 0, 0)
    );
    // SAFETY: valid live prompt can be submitted after admission rejection.
    assert_eq!(
        unsafe { aie_begin_ids(h, prompt.as_ptr(), prompt.len(), 4) },
        code::OK
    );
    assert_eq!(wait(h).kind, kind::STARTED);
    close(h);
    assert_eq!(aie_next(h), code::INVALID_HANDLE);
    assert_eq!(aie_close(h), code::INVALID_HANDLE);
    let fresh = ready();
    assert_ne!(fresh, h);
    close(fresh);
}
#[test]
fn bad_buffers_and_single_worker_admission_do_not_consume_events() {
    let _serial = SERIAL.lock().unwrap();
    let h = ready();
    let mut other = 99;
    // SAFETY: readable source and valid disjoint stack output.
    assert_eq!(
        unsafe { aie_open(0, TINY.as_ptr(), TINY.len(), &mut other) },
        code::BUSY
    );
    assert_eq!(other, 99);
    // SAFETY: null buffers/oversized counts are rejected before dereference.
    unsafe {
        assert_eq!(
            aie_begin_ids(h, std::ptr::null(), 1, 4),
            code::INVALID_INPUT
        );
        assert_eq!(
            aie_begin_ids(h, std::ptr::dangling(), usize::MAX, 4),
            code::INVALID_INPUT
        );
        assert_eq!(aie_poll(h, std::ptr::null_mut()), code::INVALID_INPUT);
    }
    assert_eq!(aie_reset(h), code::OK);
    // SAFETY: null is deliberately rejected before consuming the pending event.
    assert_eq!(
        unsafe { aie_poll(h, std::ptr::null_mut()) },
        code::INVALID_INPUT
    );
    assert_eq!(wait(h).kind, kind::RESET);
    close(h);
}
#[test]
fn initialization_failure_and_worker_panic_are_terminal() {
    let _serial = SERIAL.lock().unwrap();
    let bad = b"not model JSON";
    let mut h = 0;
    // SAFETY: bad content is still a valid readable buffer; output is live.
    assert_eq!(
        unsafe { aie_open(0, bad.as_ptr(), bad.len(), &mut h) },
        code::OK
    );
    assert_eq!(wait(h).code, code::MODEL_FAILED);
    assert_eq!(aie_next(h), code::EXECUTION_FAILED);
    close(h);
    let h = ready();
    assert_eq!(submit(h, Command::Panic), Ok(()));
    let failed = wait(h);
    assert_eq!(
        (failed.kind, failed.code, failed.processed, failed.history),
        (kind::ERROR, code::PANIC, 0, 0)
    );
    assert_eq!(aie_next(h), code::EXECUTION_FAILED);
    close(h);
}

#[test]
fn close_during_inflight_work_reserves_memory_until_actual_release() {
    let _serial = SERIAL.lock().unwrap();
    let h = ready();
    let entered = Arc::new(std::sync::Barrier::new(2));
    let release = Arc::new(std::sync::Barrier::new(2));
    assert_eq!(
        submit(h, Command::Pause(entered.clone(), release.clone())),
        Ok(())
    );
    entered.wait(); // Deterministically owns a live model on paused worker.
    assert_eq!(aie_close(h), code::OK);
    assert_eq!(aie_next(h), code::INVALID_HANDLE);
    let mut other = 99;
    // SAFETY: valid readable fixture and unique writable u64.
    assert_eq!(
        unsafe { aie_open(0, TINY.as_ptr(), TINY.len(), &mut other) },
        code::BUSY
    );
    assert_eq!(other, 99);
    release.wait();
    let deadline = Instant::now() + Duration::from_secs(5);
    while WORKER_ACTIVE.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    let fresh = ready();
    assert_ne!(fresh, h);
    close(fresh);
}

#[test]
fn post_kv_write_numerical_failure_has_no_valid_prefix_or_resume() {
    let _serial = SERIAL.lock().unwrap();
    // Hand-built finite model. Q/K=0; V identity writes KV successfully;
    // SwiGLU down multiplication overflows afterward, so all KV invalidates.
    let source = serde_json::json!({
        "config": {"vocab_size":2,"layers":1,"hidden_dim":2,"query_heads":1,
            "kv_heads":1,"head_dim":2,"ff_dim":2,"max_tokens":2,"rope_base":10000.,"norm_epsilon":1e-5},
        "weights": {"embedding":[0.,0.,1.,1.],"final_norm":[1.,1.],"layers":[{
            "attention_norm":[1.,1.],"query":([0.;4].to_vec()),"key":([0.;4].to_vec()),
            "value":[1.,0.,0.,1.],"attention_output":([0.;4].to_vec()),
            "ffn_norm":[1.,1.],"gate":([1.;4].to_vec()),"up":([1.;4].to_vec()),"down":([f32::MAX;4].to_vec())
        }]}
    }).to_string();
    let mut h = 0;
    // SAFETY: source lives through bounded copy, output is a disjoint stack u64.
    assert_eq!(
        unsafe { aie_open(0, source.as_ptr(), source.len(), &mut h) },
        code::OK
    );
    assert_eq!(wait(h).kind, kind::READY);
    let ids = [1_u32];
    // SAFETY: one aligned readable ID, copied during call.
    assert_eq!(unsafe { aie_begin_ids(h, ids.as_ptr(), 1, 1) }, code::OK);
    let failed = wait(h);
    assert_eq!(
        (failed.kind, failed.code, failed.processed, failed.history),
        (kind::ERROR, code::EXECUTION_FAILED, 0, 1)
    );
    assert_eq!(aie_next(h), code::EXECUTION_FAILED);
    assert_eq!(aie_reset(h), code::EXECUTION_FAILED);
    close(h);
}

#[test]
fn text_poll_is_atomic_bounded_and_preserves_invalid_outputs() {
    let _serial = SERIAL.lock().unwrap();
    let h = ready();
    let ids = [1, 5, 7];
    // SAFETY: bounded readable aligned IDs.
    assert_eq!(unsafe { aie_begin_ids(h, ids.as_ptr(), 3, 4) }, code::OK);
    assert_eq!(wait(h).kind, kind::STARTED);
    assert_eq!(aie_next(h), code::OK);
    let mut out = event();
    let mut bytes = [99_u8; MAX_TEXT];
    let mut length = 99;
    // SAFETY: all outputs are disjoint live allocations; invalid capacity is
    // rejected before reads/writes or event consumption.
    assert_eq!(
        unsafe { aie_poll_text(h, &mut out, bytes.as_mut_ptr(), MAX_TEXT - 1, &mut length) },
        code::INVALID_INPUT
    );
    assert_eq!(out, event());
    assert_eq!(bytes, [99; MAX_TEXT]);
    assert_eq!(length, 99);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        // SAFETY: disjoint writable Event/usize and 4096-byte allocation.
        let code = unsafe { aie_poll_text(h, &mut out, bytes.as_mut_ptr(), MAX_TEXT, &mut length) };
        if code == code::OK {
            break;
        }
        assert_eq!(code, code::PENDING);
        assert_eq!(out, event());
        assert_eq!(length, 99);
        assert_eq!(bytes, [99; MAX_TEXT]);
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(out.token, 41);
    assert_eq!(&bytes[..length], b"41 ");
    assert!(bytes[length..].iter().all(|&b| b == 99));
    close(h);
}

#[test]
fn cancellation_before_prompt_admission_can_clear_and_retry() {
    let _serial = SERIAL.lock().unwrap();
    let h = ready();
    assert_eq!(aie_set_cancel(h, 1), code::OK);
    let ids = [1, 5, 7];
    // SAFETY: bounded aligned readable IDs copied during call.
    assert_eq!(unsafe { aie_begin_ids(h, ids.as_ptr(), 3, 4) }, code::OK);
    let cancelled = wait(h);
    assert_eq!(
        (cancelled.kind, cancelled.processed, cancelled.history),
        (kind::CANCELLED, 0, 0)
    );
    assert_eq!(aie_set_cancel(h, 0), code::OK);
    // SAFETY: same live readable IDs, previous cancellation retained no prompt.
    assert_eq!(unsafe { aie_begin_ids(h, ids.as_ptr(), 3, 4) }, code::OK);
    let admitted = wait(h);
    assert_eq!(
        (admitted.kind, admitted.processed, admitted.history),
        (kind::STARTED, 3, 3)
    );
    close(h);
}
