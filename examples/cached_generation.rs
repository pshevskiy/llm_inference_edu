use llm_inference::generation::GenerationSession;
use llm_inference::model::Decoder;
use llm_inference::sampling::Sampling;
use std::sync::atomic::AtomicBool;
fn main() {
    let model = Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap();
    for policy in [
        Sampling::Greedy,
        Sampling::Temperature {
            temperature: 0.7,
            top_k: 8,
            seed: 42,
        },
    ] {
        let mut state = GenerationSession::new(&model, 32, policy, &[], 4).unwrap();
        state.start(&[1, 5, 7]).unwrap();
        let cancel = AtomicBool::new(false);
        while let Some(id) = state.next_token(&cancel).unwrap() {
            println!(
                "{policy:?}: emitted={id}, history={}, evaluated={}",
                state.history().len(),
                state.processed()
            );
        }
        println!(
            "stop={:?}; exact token history={:?}",
            state.stop_reason(),
            state.history()
        );
    }
}
