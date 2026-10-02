use llm_inference::benchmark::{
    Mode, Workload, distribution, run_sample, scratch_estimate, throughput,
};
use llm_inference::kv_cache::CachedSession;
use llm_inference::model::Decoder;
fn model() -> Decoder {
    Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap()
}
fn workload() -> Workload {
    serde_json::from_str(include_str!("../benchmarks/tiny.json")).unwrap()
}
#[test]
fn decode_formula_and_robust_summary_have_independent_expected_values() {
    assert_eq!(throughput(1, 1000), None);
    assert_eq!(throughput(4, 0), None);
    assert_eq!(throughput(4, 1_500_000_000), Some(2.0));
    let d = distribution(&[1., 9., 3., 5.]).unwrap();
    assert_eq!(d.median, 4.);
    assert_eq!(d.median_absolute_deviation, 2.);
    assert_eq!(d.min, 1.);
    assert_eq!(d.max, 9.);
    assert!(distribution(&[]).is_err());
    assert!(distribution(&[f64::NAN]).is_err());
}
#[test]
fn fixed_throughput_keeps_pending_emitted_token_during_continuation() {
    let model = model();
    let workload = workload();
    workload.validate(model.config()).unwrap();
    let mut session = CachedSession::new(&model, 32).unwrap();
    let sample = run_sample(&mut session, &workload, Mode::TeacherForced, None).unwrap();
    assert_eq!(sample.emitted, workload.fixed_output);
    assert!(!sample.stopped_by_eos);
    assert_eq!(sample.backend_decode_ns.len(), 3);
    assert_eq!(sample.pending_emitted_tokens, 1);
    assert_eq!(sample.processed_before_continuation, 19);
    assert_eq!(sample.processed_after_continuation, 22);
    assert_eq!(sample.ttft_ns, None);
    assert_eq!(sample.pending_after_continuation, 1);
    assert!(sample.continuation_emitted_token < 64);
    assert!(sample.first_logits_ready_ns <= sample.token_times_ns[0]);
    assert!(sample.first_visible_text_ns.is_none());
    assert_eq!(sample.sampling_ns, [0; 4]);
    assert_eq!(sample.text_ns, [0; 4]);
    let mut all = workload.prompt.clone();
    all.extend(&workload.fixed_output);
    all.extend(&workload.continuation);
    all.push(1);
    assert_eq!(
        session.step(1).unwrap(),
        model.forward(&all, false).unwrap().logits
    );
}
#[test]
fn bad_workload_is_rejected_before_reset_or_writes() {
    let model = model();
    let mut session = CachedSession::new(&model, 32).unwrap();
    session.step(1).unwrap();
    let mut workload = workload();
    workload.fixed_output[2] = 64;
    assert!(run_sample(&mut session, &workload, Mode::TeacherForced, None).is_err());
    assert_eq!(session.processed(), 1);
    workload.fixed_output[2] = 41;
    workload.capacity = 16;
    assert!(workload.validate(model.config()).is_err());
    workload.capacity = 22;
    assert!(workload.validate(model.config()).is_err());
    workload.capacity = 23;
    workload.validate(model.config()).unwrap();
    workload.capacity = 32;
    workload.repetitions = 0;
    assert!(workload.validate(model.config()).is_err());
}
#[test]
fn diagnostic_scopes_preserve_exact_logits_and_state() {
    let model = model();
    let mut normal = CachedSession::new(&model, 8).unwrap();
    let mut diagnostic = CachedSession::new(&model, 8).unwrap();
    for id in [1, 5, 7] {
        let (actual, profile) = diagnostic.step_profiled(id).unwrap();
        assert_eq!(actual, normal.step(id).unwrap());
        assert_eq!(profile.layers.len(), 2);
        assert_eq!(diagnostic.processed(), normal.processed());
        assert!(profile.total_ns >= profile.logits_ns);
        assert!(profile.layers.iter().all(|l| l.total_ns >= l.qkv_ns));
    }
}
#[test]
fn memory_estimate_is_distinct_checked_vector_payload() {
    let model = model();
    assert_eq!(model.logical_weight_bytes().unwrap(), 82560);
    // Reusable workspace: 8*32+3*16+3*64+2*32=560 elements.
    // Conservative caller rows: 2*64=128 elements.
    assert_eq!(scratch_estimate(model.config(), 32).unwrap(), 688 * 4);
    let mut c = model.config();
    c.hidden_dim = usize::MAX;
    assert!(scratch_estimate(c, 32).is_err());
}
