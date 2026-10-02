//! Selected Qwen plain-message template and established byte-level BPE tokenizer.
//! Tool messages/calls are outside this initial contract.
use crate::weights::{Inventory, LoadError, verified_file};
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}
impl Role {
    fn name(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}
pub struct Message<'a> {
    pub role: Role,
    pub content: &'a str,
}
#[derive(Debug)]
pub enum TokenizerError {
    Asset(LoadError),
    Json(String),
    Encoding(String),
    UnsupportedAssets,
    EmptyMessages,
    InvalidToken(u32),
    InvalidStream,
}

/// Renders only the pinned template's no-tools, plain-content branch. Each
/// message gets exactly one start/end pair. No extra BOS/EOS is inserted.
pub fn render_chat(
    messages: &[Message<'_>],
    generation_prompt: bool,
) -> Result<String, TokenizerError> {
    if messages.is_empty() {
        return Err(TokenizerError::EmptyMessages);
    }
    let mut text = String::new();
    if messages[0].role != Role::System {
        text.push_str("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n");
    }
    for message in messages {
        text.push_str("<|im_start|>");
        text.push_str(message.role.name());
        text.push('\n');
        text.push_str(message.content);
        text.push_str("<|im_end|>\n");
    }
    if generation_prompt {
        text.push_str("<|im_start|>assistant\n");
    }
    Ok(text)
}

pub struct ChatTokenizer {
    tokenizer: Tokenizer,
    stop_tokens: Vec<usize>,
    special_tokens: Vec<u32>,
}
impl ChatTokenizer {
    pub fn load(directory: &Path, inventory: &Inventory) -> Result<Self, TokenizerError> {
        let parse = |name| -> Result<Value, TokenizerError> {
            serde_json::from_slice(
                &verified_file(directory, name, inventory).map_err(TokenizerError::Asset)?,
            )
            .map_err(|e| TokenizerError::Json(e.to_string()))
        };
        let config = parse("tokenizer_config.json")?;
        if config["tokenizer_class"] != "Qwen2Tokenizer"
            || config["add_bos_token"] != false
            || !config["bos_token"].is_null()
            || config["eos_token"] != "<|im_end|>"
        {
            return Err(TokenizerError::UnsupportedAssets);
        }
        let generation = parse("generation_config.json")?;
        let stop_tokens = generation["eos_token_id"]
            .as_array()
            .ok_or(TokenizerError::UnsupportedAssets)?
            .iter()
            .map(|id| {
                id.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or(TokenizerError::UnsupportedAssets)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if stop_tokens != [151645, 151643] {
            return Err(TokenizerError::UnsupportedAssets);
        }
        let bytes =
            verified_file(directory, "tokenizer.json", inventory).map_err(TokenizerError::Asset)?;
        let mut tokenizer =
            Tokenizer::from_bytes(bytes).map_err(|e| TokenizerError::Encoding(e.to_string()))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| TokenizerError::Encoding(e.to_string()))?;
        for (text, id) in [
            ("<|im_start|>", 151644),
            ("<|im_end|>", 151645),
            ("<|endoftext|>", 151643),
        ] {
            if tokenizer.token_to_id(text) != Some(id) {
                return Err(TokenizerError::UnsupportedAssets);
            }
        }
        let special_tokens = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(id, token)| token.special.then_some(id))
            .collect();
        Ok(Self {
            tokenizer,
            stop_tokens,
            special_tokens,
        })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| TokenizerError::Encoding(e.to_string()))?;
        Ok(encoded.get_ids().iter().map(|&id| id as usize).collect())
    }
    pub fn chat(
        &self,
        messages: &[Message<'_>],
        generation_prompt: bool,
    ) -> Result<Vec<usize>, TokenizerError> {
        self.encode(&render_chat(messages, generation_prompt)?)
    }
    pub fn decode(&self, ids: &[usize], skip_special: bool) -> Result<String, TokenizerError> {
        let ids = ids
            .iter()
            .map(|&id| {
                let id = u32::try_from(id).map_err(|_| TokenizerError::UnsupportedAssets)?;
                if self.tokenizer.id_to_token(id).is_none() {
                    return Err(TokenizerError::InvalidToken(id));
                }
                Ok(id)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.tokenizer
            .decode(&ids, skip_special)
            .map_err(|e| TokenizerError::Encoding(e.to_string()))
    }
    /// Reuses the established context-aware decoder; incomplete trailing byte
    /// tokens are withheld rather than printed as replacement characters.
    pub fn text_stream(&self) -> TextStream<'_> {
        TextStream {
            tokenizer: self,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            valid: true,
        }
    }
    pub fn stop_tokens(&self) -> &[usize] {
        &self.stop_tokens
    }
}

/// Text is only a presentation view; generation retains every original token ID.
/// Unfinished decoder tokens remain pending at cancellation/output limits.
pub struct TextStream<'a> {
    tokenizer: &'a ChatTokenizer,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    valid: bool,
}
impl TextStream<'_> {
    pub fn push(&mut self, id: usize) -> Result<Option<String>, TokenizerError> {
        if !self.valid {
            return Err(TokenizerError::InvalidStream);
        }
        let id = u32::try_from(id).map_err(|_| TokenizerError::UnsupportedAssets)?;
        if self.tokenizer.tokenizer.id_to_token(id).is_none() {
            return Err(TokenizerError::InvalidToken(id));
        }
        if self.tokenizer.special_tokens.contains(&id) {
            return Ok(None);
        }
        match tokenizers::tokenizer::step_decode_stream(
            &self.tokenizer.tokenizer,
            id,
            true,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        ) {
            Ok(text) => Ok(text),
            Err(error) => {
                self.valid = false;
                Err(TokenizerError::Encoding(error.to_string()))
            }
        }
    }
    pub fn pending_tokens(&self) -> usize {
        self.ids.len().saturating_sub(self.prefix_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_utf8_is_buffered_until_complete_and_stop_tokens_are_hidden() {
        let model: tokenizers::models::bpe::BPE =
            serde_json::from_str(r#"{"vocab":{"Ã":0,"©":1,"x":2},"merges":[]}"#).unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::default(),
        ));
        tokenizer.add_special_tokens(&[tokenizers::AddedToken::from("<stop>", true)]);
        let stop = tokenizer.token_to_id("<stop>").unwrap();
        let chat = ChatTokenizer {
            tokenizer,
            stop_tokens: vec![stop as usize],
            special_tokens: vec![stop],
        };
        let mut stream = chat.text_stream();
        assert_eq!(stream.push(0).unwrap(), None);
        assert_eq!(stream.pending_tokens(), 1);
        assert!(matches!(
            stream.push(99),
            Err(TokenizerError::InvalidToken(99))
        ));
        assert_eq!(stream.pending_tokens(), 1);
        assert_eq!(stream.push(1).unwrap(), Some("é".into()));
        assert_eq!(stream.pending_tokens(), 0);
        assert_eq!(stream.push(2).unwrap(), Some("x".into()));
        assert_eq!(stream.push(stop as usize).unwrap(), None);
        assert_eq!(stream.pending_tokens(), 0);
        assert_eq!(chat.decode(&[0, 1, 2, stop as usize], true).unwrap(), "éx");
    }
}
