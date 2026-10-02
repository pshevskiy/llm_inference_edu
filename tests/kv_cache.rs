use llm_inference::kv_cache::{CacheError, CachedSession};
use llm_inference::model::{Decoder, LayerWeights, ModelConfig, ModelWeights};
fn model() -> Decoder {
    Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap()
}
#[test]
fn every_cached_teacher_forced_position_equals_uncached_including_gqa() {
    let model = model();
    let mut session = CachedSession::new(&model, 8).unwrap();
    let ids = [1, 5, 7, 7, 41, 53, 2, 63];
    for (p, &id) in ids.iter().enumerate() {
        let cached = session.step(id).unwrap();
        let uncached = model.forward(&ids[..=p], false).unwrap().logits;
        assert_eq!(cached, uncached, "position {p}");
        assert_eq!(session.processed(), p + 1);
        assert!(session.is_valid());
    }
    assert!(matches!(session.step(1), Err(CacheError::Full)));
    assert_eq!(session.processed(), 8);
    assert!(session.is_valid());
    assert_eq!(session.logical_bytes(), 2 * 2 * 8 * 8 * 2 * 4);
}
#[test]
fn reset_reuses_storage_and_repeated_tokens_use_new_absolute_positions() {
    let model = model();
    let mut session = CachedSession::new(&model, 3).unwrap();
    let first = session.step(5).unwrap();
    let second = session.step(5).unwrap();
    assert_eq!(first, model.forward(&[5], false).unwrap().logits);
    assert_eq!(second, model.forward(&[5, 5], false).unwrap().logits);
    assert_ne!(first, second);
    let bytes = session.logical_bytes();
    session.reset();
    assert_eq!(session.processed(), 0);
    assert_eq!(session.step(5).unwrap(), first);
    assert_eq!(session.logical_bytes(), bytes);
}
#[test]
fn admission_errors_preserve_valid_prefix() {
    let model = model();
    assert!(matches!(
        CachedSession::new(&model, 0),
        Err(CacheError::InvalidCapacity)
    ));
    assert!(matches!(
        CachedSession::new(&model, 33),
        Err(CacheError::InvalidCapacity)
    ));
    let mut session = CachedSession::new(&model, 2).unwrap();
    session.step(1).unwrap();
    assert!(matches!(session.step(64), Err(CacheError::InvalidToken)));
    assert!(session.is_valid());
    assert_eq!(session.processed(), 1);
    assert_eq!(
        session.step(5).unwrap(),
        model.forward(&[1, 5], false).unwrap().logits
    );
}
#[test]
fn partial_layer_failure_invalidates_all_state_until_reset() {
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
        config,
        ModelWeights {
            embedding: (vec![1.; 4]).into(),
            layers: vec![layer],
            final_norm: vec![1.; 2],
        },
    )
    .unwrap();
    let mut session = CachedSession::new(&model, 2).unwrap();
    assert!(matches!(session.step(0), Err(CacheError::NonFinite)));
    assert!(!session.is_valid());
    assert_eq!(session.processed(), 0);
    assert!(matches!(session.step(0), Err(CacheError::InvalidState)));
    session.reset();
    assert!(session.is_valid());
    assert_eq!(session.processed(), 0);
}
#[test]
fn cached_qkv_biases_match_the_uncached_path() {
    let mut fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/tiny-model.json")).unwrap();
    for l in fixture["weights"]["layers"].as_array_mut().unwrap() {
        l["query_bias"] = serde_json::json!([0.125; 32].to_vec());
        l["key_bias"] = serde_json::json!([-0.25; 16].to_vec());
        l["value_bias"] = serde_json::json!([0.5; 16].to_vec());
    }
    let model = Decoder::from_json(&fixture.to_string()).unwrap();
    let mut session = CachedSession::new(&model, 3).unwrap();
    for (p, &id) in [1, 5, 7].iter().enumerate() {
        assert_eq!(
            session.step(id).unwrap(),
            model.forward(&[1, 5, 7][..=p], false).unwrap().logits
        );
    }
}

#[test]
fn large_logical_cache_is_rejected_before_allocation() {
    let mut fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/tiny-model.json")).unwrap();
    fixture["config"]["max_tokens"] = serde_json::Value::from(1 << 24);
    let model = Decoder::from_json(&fixture.to_string()).unwrap();
    assert!(matches!(
        CachedSession::new(&model, 1 << 24),
        Err(CacheError::ResourceBudget)
    ));
}
