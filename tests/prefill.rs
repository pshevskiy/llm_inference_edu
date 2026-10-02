use llm_inference::kv_cache::{CacheError, CachedSession};
use llm_inference::model::{Decoder, LayerWeights, ModelConfig, ModelWeights};
fn model() -> Decoder {
    Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap()
}
#[test]
fn one_shot_chunks_and_token_steps_are_identical_at_every_boundary() {
    let model = model();
    let ids = [1, 5, 7, 41, 53, 41, 50, 2];
    let expected = model.forward(&ids, false).unwrap().logits;
    let mut session = CachedSession::new(&model, 8).unwrap();
    assert_eq!(session.prefill(&ids).unwrap(), expected);
    assert_eq!(session.processed(), 8);
    for partitions in [vec![1, 7], vec![3, 5], vec![2, 2, 4], vec![1; 8]] {
        session.reset();
        let mut end = 0;
        for chunk in partitions {
            let begin = end;
            end += chunk;
            let result = if begin == 0 {
                session.prefill(&ids[begin..end])
            } else {
                session.append(&ids[begin..end])
            }
            .unwrap();
            assert_eq!(result, model.forward(&ids[..end], false).unwrap().logits);
            assert_eq!(session.processed(), end);
        }
        assert!(matches!(session.append(&[1]), Err(CacheError::Full)));
        assert_eq!(session.processed(), 8);
        assert!(session.is_valid());
    }
}
#[test]
fn entire_chunk_is_admitted_before_writes_and_prefill_requires_empty_state() {
    let model = model();
    let mut session = CachedSession::new(&model, 4).unwrap();
    assert!(matches!(
        session.prefill(&[1, 64]),
        Err(CacheError::InvalidToken)
    ));
    assert_eq!(session.processed(), 0);
    assert!(matches!(session.prefill(&[]), Err(CacheError::EmptyInput)));
    session.prefill(&[1, 5]).unwrap();
    assert!(matches!(session.prefill(&[7]), Err(CacheError::NotEmpty)));
    assert!(matches!(
        session.append(&[7, 64]),
        Err(CacheError::InvalidToken)
    ));
    assert!(matches!(session.append(&[7, 7, 7]), Err(CacheError::Full)));
    assert_eq!(session.processed(), 2);
    assert!(session.is_valid());
    assert_eq!(
        session.append(&[7, 41]).unwrap(),
        model.forward(&[1, 5, 7, 41], false).unwrap().logits
    );
}
#[test]
fn execution_failure_after_a_successful_chunk_token_invalidates_prefix() {
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
            embedding: (vec![0., 0., 1., 1.]).into(),
            layers: vec![layer],
            final_norm: vec![1.; 2],
        },
    )
    .unwrap();
    let mut session = CachedSession::new(&model, 2).unwrap();
    assert!(matches!(
        session.prefill(&[0, 1]),
        Err(CacheError::NonFinite)
    ));
    assert_eq!(session.processed(), 0);
    assert!(!session.is_valid());
    assert!(matches!(
        session.append(&[0]),
        Err(CacheError::InvalidState)
    ));
    session.reset();
    assert_eq!(session.prefill(&[0]).unwrap(), [0., 0.]);
}
