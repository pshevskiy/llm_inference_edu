use llm_inference::generation::GenerationSession;
use llm_inference::kv_cache::{CacheError, CachedSession};
use llm_inference::model::Decoder;
use llm_inference::sampling::Sampling;
use serde::Deserialize;
use std::sync::atomic::AtomicBool;
#[derive(Deserialize)]
struct Row {
    position: usize,
    logits: Vec<f32>,
    greedy: usize,
}
#[derive(Deserialize)]
struct Fixture {
    ids: Vec<usize>,
    positions: Vec<Row>,
}
fn model() -> Decoder {
    let mut json: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/tiny-model.json")).unwrap();
    json["config"]["max_tokens"] = 128.into();
    Decoder::from_json(&json.to_string()).unwrap()
}
#[test]
fn larger_cache_stride_and_rotary_positions_match_independent_reference() {
    let model = model();
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/t28-tiny-context.json")).unwrap();
    let mut large = CachedSession::new(&model, 128).unwrap();
    let mut small = CachedSession::new(&model, 32).unwrap();
    let mut logits = vec![0.; 64];
    for (position, &token) in fixture.ids.iter().enumerate() {
        large.step_into(token, &mut logits).unwrap();
        if position < 32 {
            assert_eq!(small.step(token).unwrap(), logits);
        }
        if let Some(row) = fixture
            .positions
            .iter()
            .find(|row| row.position == position)
        {
            for (&actual, &expected) in logits.iter().zip(&row.logits) {
                assert!(actual.is_finite() && expected.is_finite());
                assert!((actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs());
            }
            let greedy = (0..64).fold(0, |best, i| if logits[i] > logits[best] { i } else { best });
            assert_eq!(greedy, row.greedy);
        }
        assert_eq!(large.processed(), position + 1);
    }
    let unchanged = logits.clone();
    assert!(matches!(
        large.step_into(2, &mut logits),
        Err(CacheError::Full)
    ));
    assert_eq!(logits, unchanged);
    assert_eq!(large.processed(), 128);
    assert!(large.is_valid());
    assert_eq!(large.logical_bytes(), 4 * small.logical_bytes());
}
#[test]
fn pending_token_and_tool_suffix_cross_the_old_32_position_boundary_exactly() {
    let model = model();
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/t28-tiny-context.json")).unwrap();
    let cancel = AtomicBool::new(false);
    let mut state = GenerationSession::new(&model, 128, Sampling::Greedy, &[], 2).unwrap();
    state.start(&fixture.ids[..30]).unwrap();
    state.next_token(&cancel).unwrap();
    state.next_token(&cancel).unwrap();
    assert_eq!(state.processed(), 31);
    assert_eq!(state.history().len(), 32);
    state
        .append_context(&fixture.ids[32..36], 2, &cancel)
        .unwrap();
    assert_eq!(state.processed(), 36);
    assert_eq!(state.history().len(), 36);
    let expected = model.forward(state.history(), false).unwrap().logits;
    let greedy = (0..64).fold(0, |best, i| {
        if expected[i] > expected[best] {
            i
        } else {
            best
        }
    });
    assert_eq!(state.next_token(&cancel).unwrap().unwrap(), greedy);
    assert_eq!(state.processed(), 36);
    assert_eq!(state.history().len(), 37);
}
