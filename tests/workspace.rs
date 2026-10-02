//! One test in this process avoids overlapping allocator regions from parallel tests.
use llm_inference::generation::GenerationSession;
use llm_inference::kv_cache::{CacheError, CachedSession};
use llm_inference::model::{Decoder, LayerWeights, ModelConfig, ModelWeights};
use llm_inference::sampling::Sampling;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
use std::sync::atomic::AtomicBool;
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
fn no_allocations(operation: impl FnOnce()) {
    let region = Region::new(GLOBAL);
    operation();
    let stats = region.change();
    assert_eq!(stats.allocations, 0);
    assert_eq!(stats.reallocations, 0);
}
#[test]
fn buffered_decode_allocation_admission_reset_and_failure_contracts() {
    let model = Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap();
    let mut session = CachedSession::new(&model, 8).unwrap();
    let mut logits = vec![123.0; 64];
    let pointer = logits.as_ptr();
    let bytes = session.workspace_bytes();
    assert_eq!(bytes, (8 * 32 + 3 * 16 + 3 * 64 + 2 * 8) * 4);
    let ids = [1, 5, 7, 7, 41, 53, 2, 63];
    for _ in 0..4 {
        no_allocations(|| session.reset());
        for (p, &id) in ids.iter().enumerate() {
            no_allocations(|| session.step_into(id, &mut logits).unwrap());
            assert_eq!(logits, model.forward(&ids[..=p], false).unwrap().logits);
            assert_eq!(logits.as_ptr(), pointer);
            assert_eq!(session.workspace_bytes(), bytes);
        }
    }
    let last = logits.clone();
    assert!(matches!(
        session.step_into(1, &mut logits),
        Err(CacheError::Full)
    ));
    assert_eq!(logits, last);
    session.reset();
    assert!(matches!(
        session.step_into(64, &mut logits),
        Err(CacheError::InvalidToken)
    ));
    assert!(matches!(
        session.step_into(1, &mut logits[..63]),
        Err(CacheError::InvalidOutputLength)
    ));
    assert!(matches!(
        session.append_into(&[1, 64], &mut logits),
        Err(CacheError::InvalidToken)
    ));
    assert_eq!(session.processed(), 0);
    assert_eq!(logits, last);
    no_allocations(|| session.prefill_into(&ids[..3], &mut logits).unwrap());
    no_allocations(|| session.append_into(&ids[3..], &mut logits).unwrap());
    assert_eq!(logits, model.forward(&ids, false).unwrap().logits);
    let mut generation = GenerationSession::new(&model, 32, Sampling::Greedy, &[], 4).unwrap();
    let cancel = AtomicBool::new(false);
    for _ in 0..3 {
        no_allocations(|| {
            generation.reset();
            generation.start(&[1, 5, 7]).unwrap();
            for expected in [41, 53, 41, 50] {
                assert_eq!(generation.next_token(&cancel).unwrap(), Some(expected));
            }
        });
        assert_eq!(generation.processed(), 6);
        assert_eq!(generation.history(), [1, 5, 7, 41, 53, 41, 50]);
    }
    // A token that overflows after KV writes must invalidate; stale workspaces
    // must be fully overwritten when reset admits a different, valid zero token.
    let config = ModelConfig {
        vocab_size: 2,
        layers: 1,
        hidden_dim: 2,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 2,
        ff_dim: 2,
        max_tokens: 2,
        rope_base: 10000.,
        norm_epsilon: 1e-5,
    };
    let layer = LayerWeights {
        attention_norm: vec![1.; 2],
        query: vec![0.; 4],
        key: vec![0.; 4],
        value: vec![1., 0., 0., 1.],
        attention_output: vec![0.; 4],
        query_bias: None,
        key_bias: None,
        value_bias: None,
        ffn_norm: vec![1.; 2],
        gate: vec![1.; 4],
        up: vec![1.; 4],
        down: vec![f32::MAX; 4],
    };
    let fragile = Decoder::new(
        config,
        ModelWeights {
            embedding: vec![0., 0., 1., 1.],
            layers: vec![layer],
            final_norm: vec![1.; 2],
        },
    )
    .unwrap();
    let mut cache = CachedSession::new(&fragile, 2).unwrap();
    let mut out = [17.; 2];
    assert!(matches!(
        cache.step_into(1, &mut out),
        Err(CacheError::NonFinite)
    ));
    assert!(!cache.is_valid());
    assert_eq!(cache.processed(), 0);
    assert!(matches!(
        cache.step_into(0, &mut out),
        Err(CacheError::InvalidState)
    ));
    no_allocations(|| {
        cache.reset();
        cache.step_into(0, &mut out).unwrap();
    });
    assert_eq!(out, [0.; 2]);
    assert!(cache.is_valid());
    assert_eq!(cache.processed(), 1);
}
