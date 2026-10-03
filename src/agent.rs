//! One bounded calculator protocol. Model text is parsed as data, never executed.
//! Transcript serialization is independent of volatile KV storage; no automatic
//! filesystem persistence or private prompt logging. No general tool framework.
use crate::generation::{GenerationSession, StateError};
use crate::model::Decoder;
use crate::sampling::Sampling;
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;

pub const MAX_REPLY_BYTES: usize = 4096;
pub const MAX_TOOL_CALLS: usize = 8;
pub const MAX_TRANSCRIPT_BYTES: usize = 65536;

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalculatorArgs {
    pub operation: Operation,
    pub left: f64,
    pub right: f64,
}
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolName {
    Calculator,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Reply {
    Tool {
        name: ToolName,
        arguments: CalculatorArgs,
    },
    Final {
        text: String,
    },
}
#[derive(Debug)]
pub enum AgentError {
    InvalidLimits,
    WrongPhase,
    InvalidReply,
    PermissionDenied,
    StepLimit,
    Arithmetic,
    Generation(StateError),
    TranscriptLimit,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    New,
    Generating,
    AwaitingReply,
    PreparedResult,
    Finished,
    Failed,
}
#[derive(Debug, Serialize)]
pub struct ToolResult {
    pub text: String,
    pub token_ids: Vec<usize>,
}
#[derive(Debug, Serialize)]
pub struct Turn {
    pub generated_ids: Vec<usize>,
    pub reply: Reply,
    pub tool_result: Option<ToolResult>,
}
#[derive(Debug, Default, Serialize)]
pub struct Transcript {
    pub prompt_ids: Vec<usize>,
    pub turns: Vec<Turn>,
    /// Emitted IDs also survive invalid replies/cancellation before continuation.
    pub pending_generated_ids: Vec<usize>,
}

fn parse_reply(text: &str) -> Result<Reply, AgentError> {
    if text.is_empty() || text.len() > MAX_REPLY_BYTES {
        return Err(AgentError::InvalidReply);
    }
    serde_json::from_str(text).map_err(|_| AgentError::InvalidReply)
}
fn calculate(args: &CalculatorArgs) -> Result<String, AgentError> {
    // Concrete small calculator range; JSON parser additionally rejects NaN/Inf.
    if !args.left.is_finite()
        || !args.right.is_finite()
        || args.left.abs() > 1e12
        || args.right.abs() > 1e12
    {
        return Err(AgentError::Arithmetic);
    }
    let value = match args.operation {
        Operation::Add => args.left + args.right,
        Operation::Subtract => args.left - args.right,
        Operation::Multiply => args.left * args.right,
        Operation::Divide if args.right != 0.0 => args.left / args.right,
        Operation::Divide => return Err(AgentError::Arithmetic),
    };
    if !value.is_finite() {
        return Err(AgentError::Arithmetic);
    }
    // A trusted envelope contains only an arithmetic number, no instructions.
    let text = serde_json::json!({"calculator_result": value}).to_string();
    if text.len() > 256 {
        return Err(AgentError::Arithmetic);
    }
    Ok(text)
}

/// One model, one sequence, serial calculator calls. The caller supplies decoded
/// model replies and tokenizes the returned trusted result envelope. Neither is
/// assumed to be a demonstrated model capability: deterministic mocks are useful
/// tests but must be labelled. Editing history requires a fresh AgentSession;
/// token history/cache cannot be mutated through this API.
pub struct AgentSession<'a> {
    generation: GenerationSession<'a>,
    transcript: Transcript,
    phase: Phase,
    max_new: usize,
    max_calls: usize,
    calls: usize,
    prepared: Option<(Reply, String)>,
}
impl<'a> AgentSession<'a> {
    pub fn new(
        model: &'a Decoder,
        capacity: usize,
        max_new: usize,
        max_calls: usize,
    ) -> Result<Self, AgentError> {
        if max_calls == 0 || max_calls > MAX_TOOL_CALLS || capacity > 2048 {
            return Err(AgentError::InvalidLimits);
        }
        Ok(Self {
            generation: GenerationSession::new(model, capacity, Sampling::Greedy, &[], max_new)
                .map_err(AgentError::Generation)?,
            transcript: Transcript::default(),
            phase: Phase::New,
            max_new,
            max_calls,
            calls: 0,
            prepared: None,
        })
    }
    pub fn start(&mut self, prompt: &[usize], cancel: &AtomicBool) -> Result<(), AgentError> {
        if self.phase != Phase::New {
            return Err(AgentError::WrongPhase);
        }
        let result = self.generation.start_with_cancel(prompt, cancel);
        if !self.generation.history().is_empty() {
            self.transcript.prompt_ids = prompt.to_vec();
            self.phase = if self.generation.is_valid() {
                Phase::Generating
            } else {
                Phase::Failed
            };
        }
        result.map_err(AgentError::Generation)
    }
    pub fn next_token(&mut self, cancel: &AtomicBool) -> Result<usize, AgentError> {
        if self.phase != Phase::Generating {
            return Err(AgentError::WrongPhase);
        }
        match self.generation.next_token(cancel) {
            Ok(Some(id)) => {
                self.transcript.pending_generated_ids.push(id);
                if self.generation.stop_reason().is_some() {
                    self.phase = Phase::AwaitingReply;
                }
                Ok(id)
            }
            Ok(None) => Err(AgentError::WrongPhase),
            Err(error) => {
                if !self.generation.is_valid() {
                    self.phase = Phase::Failed;
                }
                Err(AgentError::Generation(error))
            }
        }
    }
    /// Validate the complete bounded model response before arithmetic. Permission
    /// is checked on each call; denied/invalid calls leave transcript/cache alone.
    /// Returns Some(trusted result text) or None for an accepted final answer.
    pub fn prepare_reply(
        &mut self,
        text: &str,
        calculator_allowed: bool,
    ) -> Result<Option<&str>, AgentError> {
        if self.phase != Phase::AwaitingReply {
            return Err(AgentError::WrongPhase);
        }
        let reply = parse_reply(text)?;
        match &reply {
            Reply::Final { .. } => {
                self.transcript.turns.push(Turn {
                    generated_ids: std::mem::take(&mut self.transcript.pending_generated_ids),
                    reply,
                    tool_result: None,
                });
                self.phase = Phase::Finished;
                Ok(None)
            }
            Reply::Tool { arguments, .. } => {
                if !calculator_allowed {
                    return Err(AgentError::PermissionDenied);
                }
                if self.calls == self.max_calls {
                    return Err(AgentError::StepLimit);
                }
                let output = calculate(arguments)?;
                self.prepared = Some((reply, output));
                self.phase = Phase::PreparedResult;
                Ok(self.prepared.as_ref().map(|(_, text)| text.as_str()))
            }
        }
    }
    /// Caller must encode exactly the trusted text returned by prepare_reply,
    /// with its selected explicit framing. Tiny mocks use fixture IDs, not a
    /// claimed tokenizer. Invalid IDs/budget/pre-cancel preserve PreparedResult.
    /// An admitted partial cancellation records the whole suffix for exact resume.
    pub fn append_result(
        &mut self,
        encoded_result: &[usize],
        cancel: &AtomicBool,
    ) -> Result<(), AgentError> {
        if self.phase != Phase::PreparedResult {
            return Err(AgentError::WrongPhase);
        }
        let before = self.generation.history().len();
        let result = self
            .generation
            .append_context(encoded_result, self.max_new, cancel);
        if self.generation.history().len() > before {
            let (reply, text) = self
                .prepared
                .take()
                .expect("phase requires prepared result");
            self.transcript.turns.push(Turn {
                generated_ids: std::mem::take(&mut self.transcript.pending_generated_ids),
                reply,
                tool_result: Some(ToolResult {
                    text,
                    token_ids: encoded_result.to_vec(),
                }),
            });
            self.calls += 1;
            self.phase = if self.generation.is_valid() {
                Phase::Generating
            } else {
                Phase::Failed
            };
        }
        result.map_err(AgentError::Generation)
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn history(&self) -> &[usize] {
        self.generation.history()
    }
    pub fn processed(&self) -> usize {
        self.generation.processed()
    }
    pub fn transcript(&self) -> &Transcript {
        &self.transcript
    }
    /// Caller may persist this bounded JSON separately from cache. It contains
    /// prompt IDs and model text and must be treated as private user data.
    pub fn transcript_json(&self) -> Result<String, AgentError> {
        let text =
            serde_json::to_string(&self.transcript).map_err(|_| AgentError::TranscriptLimit)?;
        if text.len() > MAX_TRANSCRIPT_BYTES {
            return Err(AgentError::TranscriptLimit);
        }
        Ok(text)
    }
}
