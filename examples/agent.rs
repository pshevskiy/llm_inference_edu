//! Deterministic decision fixtures plus real tiny Rust generation. No claim that
//! the synthetic model's IDs encode JSON or demonstrate real tool capability.
use llm_inference::agent::AgentSession;
use llm_inference::model::Decoder;
use std::sync::atomic::AtomicBool;
fn main() {
    let model = Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap();
    let cancel = AtomicBool::new(false);
    let mut agent = AgentSession::new(&model, 32, 1, 1).unwrap();
    agent.start(&[1, 5, 7], &cancel).unwrap();
    println!(
        "Mock protocol; actual tiny token {} at prefix {}/{}",
        agent.next_token(&cancel).unwrap(),
        agent.processed(),
        agent.history().len()
    );
    let request =
        r#"{"kind":"tool","name":"calculator","arguments":{"operation":"add","left":2,"right":3}}"#;
    println!(
        "Trusted result: {}",
        agent.prepare_reply(request, true).unwrap().unwrap()
    );
    // Deterministic fixture encoding of tool result, not claimed real tokenization.
    agent.append_result(&[2, 3], &cancel).unwrap();
    println!(
        "After append: prefix {}/{}",
        agent.processed(),
        agent.history().len()
    );
    println!(
        "Next actual tiny token: {}",
        agent.next_token(&cancel).unwrap()
    );
    agent
        .prepare_reply(r#"{"kind":"final","text":"5"}"#, false)
        .unwrap();
    println!(
        "Caller-owned transcript (KV excluded): {}",
        agent.transcript_json().unwrap()
    );
}
