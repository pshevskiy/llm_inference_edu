use llm_inference::generation::{GenerationSession, StateError, StopReason};
use llm_inference::model::Decoder;
use llm_inference::sampling::Sampling;
use std::sync::atomic::{AtomicBool, Ordering};
fn model() -> Decoder {
    Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap()
}
#[test]
fn emitted_tokens_and_processed_prefix_are_distinct_until_the_next_step() {
    let model = model();
    let mut state = GenerationSession::new(&model, 32, Sampling::Greedy, &[], 4).unwrap();
    state.start(&[1, 5, 7]).unwrap();
    let cancel = AtomicBool::new(false);
    for (i, id) in [41, 53, 41, 50].into_iter().enumerate() {
        assert_eq!(state.next_token(&cancel).unwrap(), Some(id));
        assert_eq!(state.history().len(), 4 + i);
        assert_eq!(state.processed(), 3 + i);
    }
    assert_eq!(state.stop_reason(), Some(StopReason::OutputLimit));
    assert_eq!(state.next_token(&cancel).unwrap(), None);
    assert_eq!(state.processed(), 6);
    assert_eq!(state.history(), [1, 5, 7, 41, 53, 41, 50]);
}
#[test]
fn cancellation_preserves_pending_tokens_and_resume_does_not_duplicate_them() {
    let model = model();
    let mut state = GenerationSession::new(&model, 32, Sampling::Greedy, &[], 4).unwrap();
    state.start(&[1, 5, 7]).unwrap();
    let cancel = AtomicBool::new(true);
    assert!(matches!(
        state.next_token(&cancel),
        Err(StateError::Cancelled)
    ));
    assert_eq!(state.processed(), 3);
    assert_eq!(state.history(), [1, 5, 7]);
    cancel.store(false, Ordering::Relaxed);
    assert_eq!(state.next_token(&cancel).unwrap(), Some(41));
    cancel.store(true, Ordering::Relaxed);
    assert!(matches!(
        state.next_token(&cancel),
        Err(StateError::Cancelled)
    ));
    assert_eq!(state.history(), [1, 5, 7, 41]);
    assert_eq!(state.processed(), 3);
    assert!(state.is_valid());
    cancel.store(false, Ordering::Relaxed);
    assert_eq!(state.next_token(&cancel).unwrap(), Some(53));
    assert_eq!(state.history(), [1, 5, 7, 41, 53]);
    assert_eq!(state.processed(), 4);
}
#[test]
fn stop_reset_and_admission_limits_are_explicit() {
    let model = model();
    let mut state = GenerationSession::new(&model, 32, Sampling::Greedy, &[41], 4).unwrap();
    assert!(matches!(
        state.next_token(&AtomicBool::new(false)),
        Err(StateError::NotStarted)
    ));
    assert!(matches!(
        state.start(&[64]),
        Err(StateError::InvalidParameters)
    ));
    assert!(matches!(
        state.start(&[1; 29]),
        Err(StateError::InvalidParameters)
    ));
    state.start(&[1, 5, 7]).unwrap();
    assert!(matches!(state.start(&[1]), Err(StateError::AlreadyStarted)));
    assert_eq!(state.next_token(&AtomicBool::new(false)).unwrap(), Some(41));
    assert_eq!(state.stop_reason(), Some(StopReason::StopToken));
    assert_eq!(state.processed(), 3);
    assert_eq!(state.next_token(&AtomicBool::new(false)).unwrap(), None);
    state.reset();
    assert!(state.history().is_empty());
    state.start(&[1, 5, 7]).unwrap();
    assert_eq!(state.next_token(&AtomicBool::new(false)).unwrap(), Some(41));
    let mut exact = GenerationSession::new(&model, 32, Sampling::Greedy, &[], 1).unwrap();
    exact.start(&[1; 31]).unwrap();
    exact.next_token(&AtomicBool::new(false)).unwrap();
    assert_eq!(exact.history().len(), 32);
    assert_eq!(exact.processed(), 31);
}

#[test]
fn cancellation_before_prefill_keeps_the_session_empty_and_retryable() {
    let model = model();
    let mut state = GenerationSession::new(&model, 32, Sampling::Greedy, &[], 1).unwrap();
    let cancel = AtomicBool::new(true);
    assert!(matches!(
        state.start_with_cancel(&[1, 5, 7], &cancel),
        Err(StateError::Cancelled)
    ));
    assert!(state.history().is_empty());
    assert_eq!(state.processed(), 0);
    assert!(state.is_valid());
    cancel.store(false, Ordering::Relaxed);
    state.start_with_cancel(&[1, 5, 7], &cancel).unwrap();
    assert_eq!(state.next_token(&cancel).unwrap(), Some(41));
}

#[test]
fn partial_prefill_failure_preserves_ids_but_invalidates_execution_state() {
    use llm_inference::kv_cache::CacheError;
    use llm_inference::model::{LayerWeights, ModelConfig, ModelWeights};
    let c = ModelConfig {
        vocab_size: 2,
        layers: 1,
        hidden_dim: 2,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 2,
        ff_dim: 2,
        max_tokens: 4,
        rope_base: 10000.,
        norm_epsilon: 1e-5,
    };
    let layer = LayerWeights {
        attention_norm: vec![1.; 2],
        query: (vec![0.; 4]).into(),
        key: (vec![0.; 4]).into(),
        value: (vec![1., 0., 0., 1.]).into(),
        attention_output: (vec![0.; 4]).into(),
        query_bias: None,
        key_bias: None,
        value_bias: None,
        ffn_norm: vec![1.; 2],
        gate: (vec![1.; 4]).into(),
        up: (vec![1.; 4]).into(),
        down: (vec![f32::MAX; 4]).into(),
    };
    let model = Decoder::new(
        c,
        ModelWeights {
            embedding: (vec![0., 0., 1., 1.]).into(),
            layers: vec![layer],
            final_norm: vec![1.; 2],
        },
    )
    .unwrap();
    let mut state = GenerationSession::new(&model, 4, Sampling::Greedy, &[], 1).unwrap();
    assert!(matches!(
        state.start(&[0, 1]),
        Err(StateError::Cache(CacheError::NonFinite))
    ));
    assert_eq!(state.history(), [0, 1]);
    assert_eq!(state.processed(), 0);
    assert!(!state.is_valid());
    assert!(matches!(
        state.next_token(&AtomicBool::new(false)),
        Err(StateError::Cache(CacheError::InvalidState))
    ));
    state.reset();
    state.start(&[0]).unwrap();
    assert_eq!(state.next_token(&AtomicBool::new(false)).unwrap(), Some(0));
    // Continuation evaluates pending 0, then token1 fails after KV writes.
    assert!(matches!(
        state.append_context(&[1], 1, &AtomicBool::new(false)),
        Err(StateError::Cache(CacheError::NonFinite))
    ));
    assert_eq!(state.history(), [0, 0, 1]);
    assert_eq!(state.processed(), 0);
    assert!(!state.is_valid());
    assert!(matches!(
        state.next_token(&AtomicBool::new(false)),
        Err(StateError::Cache(CacheError::InvalidState))
    ));
    state
        .replay_history(&[0], 1, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(state.next_token(&AtomicBool::new(false)).unwrap(), Some(0));
}

#[test]
fn continuation_admission_rejects_before_writes_and_editing_replays() {
    let model = model();
    let cancel = AtomicBool::new(false);
    let mut state = GenerationSession::new(&model, 32, Sampling::Greedy, &[], 1).unwrap();
    assert!(matches!(
        state.append_context(&[2], 1, &cancel),
        Err(StateError::InvalidParameters)
    ));
    state.start(&[1, 5, 7]).unwrap();
    assert!(matches!(
        state.append_context(&[2], 1, &cancel),
        Err(StateError::InvalidParameters)
    ));
    state.next_token(&cancel).unwrap();
    for (suffix, max_new) in [
        (&[][..], 1),
        (&[64][..], 1),
        (&[1; 32][..], 1),
        (&[2][..], 0),
        (&[2][..], usize::MAX),
    ] {
        assert!(matches!(
            state.append_context(suffix, max_new, &cancel),
            Err(StateError::InvalidParameters)
        ));
        assert_eq!(state.history(), [1, 5, 7, 41]);
        assert_eq!(state.processed(), 3);
        assert_eq!(state.generated(), 1);
        assert_eq!(state.stop_reason(), Some(StopReason::OutputLimit));
    }
    for (ids, max_new) in [
        (&[][..], 1),
        (&[64][..], 1),
        (&[1; 32][..], 1),
        (&[2][..], 0),
        (&[2][..], usize::MAX),
    ] {
        assert!(matches!(
            state.replay_history(ids, max_new, &cancel),
            Err(StateError::InvalidParameters)
        ));
        assert_eq!(state.history(), [1, 5, 7, 41]);
        assert_eq!(state.processed(), 3);
    }
    cancel.store(true, Ordering::Relaxed);
    assert!(matches!(
        state.append_context(&[2, 3], 1, &cancel),
        Err(StateError::Cancelled)
    ));
    assert!(matches!(
        state.replay_history(&[2, 5, 7, 41, 2, 3], 1, &cancel),
        Err(StateError::Cancelled)
    ));
    assert_eq!(state.history(), [1, 5, 7, 41]);
    assert_eq!(state.processed(), 3);
    cancel.store(false, Ordering::Relaxed);
    state.append_context(&[2, 3], 1, &cancel).unwrap();
    assert_eq!(state.history(), [1, 5, 7, 41, 2, 3]);
    assert_eq!(state.processed(), 6);
    assert_eq!(state.next_token(&cancel).unwrap(), Some(14));
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/t25-continuation.json")).unwrap();
    let edited: Vec<usize> = expected["cases"][1]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    state.replay_history(&edited, 1, &cancel).unwrap();
    assert_eq!(state.history(), edited);
    assert_eq!(state.processed(), 6);
    assert_eq!(
        state.next_token(&cancel).unwrap().unwrap(),
        expected["cases"][1]["greedy"].as_u64().unwrap() as usize
    );
}
