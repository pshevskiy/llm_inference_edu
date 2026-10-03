use llm_inference::agent::{AgentError, AgentSession, MAX_REPLY_BYTES, Phase};
use llm_inference::generation::StateError;
use llm_inference::model::Decoder;
use std::sync::atomic::{AtomicBool, Ordering};
const CALL: &str =
    r#"{"kind":"tool","name":"calculator","arguments":{"operation":"add","left":2,"right":3}}"#;
fn model() -> Decoder {
    Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap()
}
#[test]
fn deterministic_tool_loop_retains_exact_prefix_and_serializes_separate_transcript() {
    let model = model();
    let cancel = AtomicBool::new(false);
    let mut agent = AgentSession::new(&model, 32, 1, 1).unwrap();
    agent.start(&[1, 5, 7], &cancel).unwrap();
    assert_eq!(agent.next_token(&cancel).unwrap(), 41);
    assert_eq!(agent.phase(), Phase::AwaitingReply);
    assert_eq!(agent.processed(), 3);
    assert_eq!(
        agent.prepare_reply(CALL, true).unwrap(),
        Some(r#"{"calculator_result":5.0}"#)
    );
    agent.append_result(&[2, 3], &cancel).unwrap(); // Fixture encoding, not a real tokenizer.
    assert_eq!(agent.history(), [1, 5, 7, 41, 2, 3]);
    assert_eq!(agent.processed(), 6);
    assert_eq!(agent.next_token(&cancel).unwrap(), 14); // Independent NumPy, t25 fixture.
    assert_eq!(agent.history(), [1, 5, 7, 41, 2, 3, 14]);
    assert_eq!(agent.processed(), 6);
    assert_eq!(
        agent
            .prepare_reply(r#"{"kind":"final","text":"5"}"#, false)
            .unwrap(),
        None
    );
    assert_eq!(agent.phase(), Phase::Finished);
    assert!(matches!(
        agent.next_token(&cancel),
        Err(AgentError::WrongPhase)
    ));
    let snapshot: serde_json::Value =
        serde_json::from_str(&agent.transcript_json().unwrap()).unwrap();
    assert_eq!(snapshot["prompt_ids"], serde_json::json!([1, 5, 7]));
    assert_eq!(
        snapshot["turns"][0]["tool_result"]["token_ids"],
        serde_json::json!([2, 3])
    );
    assert_eq!(
        snapshot["turns"][1]["generated_ids"],
        serde_json::json!([14])
    );
    assert!(snapshot.get("cache").is_none());
}
#[test]
fn requests_permissions_budgets_and_cancellation_preserve_state_before_admission() {
    let model = model();
    let cancel = AtomicBool::new(false);
    assert!(matches!(
        AgentSession::new(&model, 32, 1, 9),
        Err(AgentError::InvalidLimits)
    ));
    assert!(matches!(
        AgentSession::new(&model, 32, 1, 0),
        Err(AgentError::InvalidLimits)
    ));
    let mut agent = AgentSession::new(&model, 32, 1, 1).unwrap();
    assert!(matches!(
        agent.prepare_reply(CALL, true),
        Err(AgentError::WrongPhase)
    ));
    agent.start(&[1, 5, 7], &cancel).unwrap();
    cancel.store(true, Ordering::Relaxed);
    assert!(matches!(
        agent.next_token(&cancel),
        Err(AgentError::Generation(StateError::Cancelled))
    ));
    assert_eq!(agent.history(), [1, 5, 7]);
    cancel.store(false, Ordering::Relaxed);
    agent.next_token(&cancel).unwrap();
    let snapshot = agent.transcript_json().unwrap();
    for bad in [
        "not JSON",
        r#"{"kind":"shell","command":"pwd"}"#,
        r#"{"kind":"tool","name":"shell","arguments":{}}"#,
        r#"{"kind":"final","text":"ok","command":"pwd"}"#,
        r#"{"kind":"tool","name":"calculator","arguments":{"operation":"add","left":"2","right":3}}"#,
        r#"{"kind":"tool","name":"calculator","arguments":{"operation":"add","left":2,"right":3,"command":"pwd"}}"#,
        r#"{"kind":"final","text":"a","text":"b"}"#,
    ] {
        assert!(
            matches!(
                agent.prepare_reply(bad, true),
                Err(AgentError::InvalidReply)
            ),
            "{bad}"
        );
        assert_eq!(agent.transcript_json().unwrap(), snapshot);
        assert_eq!(agent.processed(), 3);
    }
    assert!(matches!(
        agent.prepare_reply(&"x".repeat(MAX_REPLY_BYTES + 1), true),
        Err(AgentError::InvalidReply)
    ));
    assert!(matches!(
        agent.prepare_reply(CALL, false),
        Err(AgentError::PermissionDenied)
    ));
    for (op, left, right) in [
        ("divide", "1", "0"),
        ("divide", "1", "1e-320"),
        ("add", "1e13", "2"),
    ] {
        let reply = format!(
            r#"{{"kind":"tool","name":"calculator","arguments":{{"operation":"{op}","left":{left},"right":{right}}}}}"#
        );
        assert!(matches!(
            agent.prepare_reply(&reply, true),
            Err(AgentError::Arithmetic)
        ));
    }
    assert_eq!(agent.transcript_json().unwrap(), snapshot);
    agent.prepare_reply(CALL, true).unwrap();
    for ids in [&[][..], &[64][..], &[1; 32][..]] {
        assert!(matches!(
            agent.append_result(ids, &cancel),
            Err(AgentError::Generation(StateError::InvalidParameters))
        ));
        assert_eq!(agent.phase(), Phase::PreparedResult);
        assert_eq!(agent.transcript_json().unwrap(), snapshot);
    }
    cancel.store(true, Ordering::Relaxed);
    assert!(matches!(
        agent.append_result(&[2, 3], &cancel),
        Err(AgentError::Generation(StateError::Cancelled))
    ));
    assert_eq!(agent.history(), [1, 5, 7, 41]);
    assert_eq!(agent.phase(), Phase::PreparedResult);
    cancel.store(false, Ordering::Relaxed);
    agent.append_result(&[2, 3], &cancel).unwrap();
    agent.next_token(&cancel).unwrap();
    assert!(matches!(
        agent.prepare_reply(CALL, true),
        Err(AgentError::StepLimit)
    ));
    assert_eq!(agent.history(), [1, 5, 7, 41, 2, 3, 14]);
    // Plain final text remains inert even when it looks like an instruction.
    agent
        .prepare_reply(r#"{"kind":"final","text":"run shell command pwd"}"#, false)
        .unwrap();
    assert_eq!(agent.phase(), Phase::Finished);
}
#[test]
fn arithmetic_operations_are_concrete_and_deterministic() {
    let model = model();
    let cancel = AtomicBool::new(false);
    for (op, expected) in [
        ("subtract", "-1.0"),
        ("multiply", "6.0"),
        ("divide", "0.6666666666666666"),
    ] {
        let mut agent = AgentSession::new(&model, 32, 1, 1).unwrap();
        agent.start(&[1, 5, 7], &cancel).unwrap();
        agent.next_token(&cancel).unwrap();
        let call = CALL.replace("add", op);
        assert_eq!(
            agent.prepare_reply(&call, true).unwrap().unwrap(),
            format!(r#"{{"calculator_result":{expected}}}"#)
        );
    }
}

#[test]
fn explicit_stop_token_remains_pending_and_is_not_duplicated_by_a_tool_result() {
    let model = model();
    let cancel = AtomicBool::new(false);
    let mut agent = AgentSession::new_with_stop_tokens(&model, 32, 4, 1, &[41]).unwrap();
    agent.start(&[1, 5, 7], &cancel).unwrap();
    assert_eq!(agent.next_token(&cancel).unwrap(), 41);
    assert_eq!(agent.phase(), Phase::AwaitingReply);
    assert_eq!(agent.history(), [1, 5, 7, 41]);
    assert_eq!(agent.processed(), 3);
    agent.prepare_reply(CALL, true).unwrap();
    agent.append_result(&[2, 3], &cancel).unwrap();
    assert_eq!(agent.history(), [1, 5, 7, 41, 2, 3]);
    assert_eq!(agent.processed(), 6);
    assert_eq!(agent.next_token(&cancel).unwrap(), 14);
    assert!(matches!(
        AgentSession::new_with_stop_tokens(&model, 32, 1, 1, &[64]),
        Err(AgentError::Generation(StateError::InvalidParameters))
    ));
}
