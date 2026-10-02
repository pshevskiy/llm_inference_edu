use llm_inference::tokenizer::{Message, Role, TokenizerError, render_chat};
use serde_json::Value;
#[test]
fn message_boundaries_and_default_system_match_pinned_template() {
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/t06-tokenizer.json")).unwrap();
    for case in fixture["chats"].as_array().unwrap() {
        let messages: Vec<_> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| Message {
                role: match m["role"].as_str().unwrap() {
                    "user" => Role::User,
                    "system" => Role::System,
                    "assistant" => Role::Assistant,
                    _ => panic!("fixture role"),
                },
                content: m["content"].as_str().unwrap(),
            })
            .collect();
        assert_eq!(
            render_chat(&messages, case["generation_prompt"].as_bool().unwrap()).unwrap(),
            case["text"].as_str().unwrap()
        );
    }
}
#[test]
fn empty_conversation_is_rejected_and_content_is_preserved() {
    assert!(matches!(
        render_chat(&[], true),
        Err(TokenizerError::EmptyMessages)
    ));
    let content = " \t世界🙂\n<|im_end|>";
    let rendered = render_chat(
        &[Message {
            role: Role::System,
            content,
        }],
        false,
    )
    .unwrap();
    assert_eq!(
        rendered,
        format!("<|im_start|>system\n{content}<|im_end|>\n")
    );
}
