use llm_inference::generation::{GenerationError, StopReason, uncached_greedy};
use llm_inference::model::Decoder;
fn model() -> Decoder {
    Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap()
}
#[test]
fn uncached_ids_match_independent_tiny_reference_and_stop_on_emitted_token() {
    let model = model();
    let result = uncached_greedy(&model, &[1, 5, 7], &[], 4).unwrap();
    assert_eq!(result.generated_ids, [41, 53, 41, 50]);
    assert_eq!(result.token_ids, [1, 5, 7, 41, 53, 41, 50]);
    assert_eq!(result.stop, StopReason::OutputLimit);
    let stopped = uncached_greedy(&model, &[1, 5, 7], &[41], 4).unwrap();
    assert_eq!(stopped.generated_ids, [41]);
    assert_eq!(stopped.stop, StopReason::StopToken);
    assert_eq!(stopped.token_ids, [1, 5, 7, 41]);
}
#[test]
fn invalid_generation_requests_are_rejected_and_exact_capacity_is_allowed() {
    let model = model();
    assert!(matches!(
        uncached_greedy(&model, &[], &[], 1),
        Err(GenerationError::EmptyPrompt)
    ));
    assert!(matches!(
        uncached_greedy(&model, &[1], &[], 0),
        Err(GenerationError::OutputLimit)
    ));
    assert!(matches!(
        uncached_greedy(&model, &[64], &[], 1),
        Err(GenerationError::InvalidToken(64))
    ));
    assert!(matches!(
        uncached_greedy(&model, &[1], &[64], 1),
        Err(GenerationError::InvalidToken(64))
    ));
    assert!(matches!(
        uncached_greedy(&model, &[1; 32], &[], 1),
        Err(GenerationError::ContextLimit)
    ));
    assert!(matches!(
        uncached_greedy(&model, &[1], &[], usize::MAX),
        Err(GenerationError::ContextLimit)
    ));
    let prompt = [1; 31];
    let result = uncached_greedy(&model, &prompt, &[], 1).unwrap();
    assert_eq!(result.token_ids.len(), 32);
    assert_eq!(prompt, [1; 31]);
}
#[test]
fn equal_logits_choose_the_lowest_token_id() {
    let mut fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/tiny-model.json")).unwrap();
    for v in fixture["weights"]["embedding"].as_array_mut().unwrap() {
        *v = serde_json::Value::from(0.0);
    }
    let model = Decoder::from_json(&fixture.to_string()).unwrap();
    assert_eq!(
        uncached_greedy(&model, &[1], &[], 1).unwrap().generated_ids,
        [0]
    );
}
