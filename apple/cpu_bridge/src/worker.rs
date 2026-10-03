//! Model, tokenizer, history, sampler and KV live on exactly one Rust worker.
//! A command produces one bounded event; the client must consume it before
//! submitting another command. No token callbacks, borrowed foreign buffers,
//! self-referential model/session object or second model/cache copy.
use crate::{Event, code, kind};
use llm_inference::generation::{GenerationSession, StateError};
use llm_inference::model::Decoder;
use llm_inference::sampling::Sampling;
use llm_inference::tokenizer::ChatTokenizer;
use llm_inference::weights::{Inventory, load_qwen};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};

pub(crate) enum Source {
    Tiny(Vec<u8>),
    Qwen(PathBuf),
}
pub(crate) enum Prompt {
    Ids(Vec<usize>),
    Text(String),
}
pub(crate) enum Command {
    Begin(Prompt, usize),
    Next,
    Reset,
    #[cfg(test)]
    Panic,
    #[cfg(test)]
    Pause(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>),
}

fn snapshot(
    generation: Option<&GenerationSession<'_>>,
    event_kind: u32,
    error: u32,
    token: Option<usize>,
) -> Event {
    Event {
        kind: event_kind,
        code: error,
        token: token.map_or(u32::MAX, |v| v as u32),
        processed: generation.map_or(0, |s| s.processed() as u32),
        history: generation.map_or(0, |s| s.history().len() as u32),
    }
}

fn failure(generation: Option<&GenerationSession<'_>>, error: StateError) -> Event {
    let error_code = match error {
        StateError::Cancelled => code::CANCELLED,
        StateError::InvalidParameters | StateError::NotStarted | StateError::AlreadyStarted => {
            code::INVALID_INPUT
        }
        _ => code::EXECUTION_FAILED,
    };
    snapshot(
        generation,
        if error_code == code::CANCELLED {
            kind::CANCELLED
        } else {
            kind::ERROR
        },
        error_code,
        None,
    )
}

pub(crate) fn run(
    source: Source,
    commands: Receiver<Command>,
    events: SyncSender<Event>,
    cancel: Arc<AtomicBool>,
    fatal: Arc<AtomicBool>,
) {
    let loaded = match source {
        Source::Tiny(bytes) => std::str::from_utf8(&bytes)
            .ok()
            .and_then(|text| Decoder::from_json(text).ok())
            .map(|m| (m, None)),
        Source::Qwen(path) => {
            let inventory: Inventory =
                serde_json::from_str(include_str!("../../../fixtures/qwen-inventory.json"))
                    .expect("embedded inventory");
            // Keep the existing short C32 CPU qualification; no automatic context expansion.
            load_qwen(&path, &inventory)
                .ok()
                .and_then(|weights| Decoder::from_qwen(weights, 32).ok())
                .and_then(|model| {
                    ChatTokenizer::load(&path, &inventory)
                        .ok()
                        .map(|t| (model, Some(t)))
                })
        }
    };
    let Some((model, tokenizer)) = loaded else {
        fatal.store(true, Ordering::Release);
        let _ = events.send(snapshot(None, kind::ERROR, code::MODEL_FAILED, None));
        return;
    };
    // Tiny JSON is caller supplied: bound allocation before building a session.
    // This bridge deliberately supports only the already-qualified small models.
    if model.config().max_tokens > 32 || model.config().vocab_size > 151936 {
        fatal.store(true, Ordering::Release);
        let _ = events.send(snapshot(None, kind::ERROR, code::MODEL_FAILED, None));
        return;
    }
    if events
        .send(snapshot(None, kind::READY, code::OK, None))
        .is_err()
    {
        return;
    }
    let mut generation: Option<GenerationSession<'_>> = None;
    let capacity = model.config().max_tokens;
    let stops = tokenizer
        .as_ref()
        .map_or(&[][..], ChatTokenizer::stop_tokens);
    while let Ok(command) = commands.recv() {
        let event = match command {
            Command::Begin(prompt, max_new) => {
                if generation.as_ref().is_some_and(|g| !g.history().is_empty()) {
                    snapshot(generation.as_ref(), kind::ERROR, code::INVALID_INPUT, None)
                } else {
                    let ids = match prompt {
                        Prompt::Ids(ids) => Some(ids),
                        Prompt::Text(text) => tokenizer.as_ref().and_then(|t| t.encode(&text).ok()),
                    };
                    match ids.and_then(|ids| {
                        if ids.is_empty()
                            || ids.iter().any(|&v| v >= model.config().vocab_size)
                            || ids.len().checked_add(max_new).is_none_or(|n| n > capacity)
                        {
                            return None;
                        }
                        Some(ids)
                    }) {
                        None => {
                            snapshot(generation.as_ref(), kind::ERROR, code::INVALID_INPUT, None)
                        }
                        Some(ids) => match GenerationSession::new(
                            &model,
                            capacity,
                            Sampling::Greedy,
                            stops,
                            max_new,
                        ) {
                            Err(error) => failure(generation.as_ref(), error),
                            Ok(mut candidate) => {
                                let result = candidate.start_with_cancel(&ids, &cancel);
                                // A partially prefetched prompt retains exact history/prefix for resume.
                                // Pre-admission cancellation preserves the previous empty session.
                                if !candidate.history().is_empty() {
                                    generation = Some(candidate);
                                }
                                match result {
                                    Ok(()) => {
                                        snapshot(generation.as_ref(), kind::STARTED, code::OK, None)
                                    }
                                    Err(error) => failure(generation.as_ref(), error),
                                }
                            }
                        },
                    }
                }
            }
            Command::Next => match &mut generation {
                None => snapshot(None, kind::ERROR, code::INVALID_INPUT, None),
                Some(g) => match g.next_token(&cancel) {
                    Ok(Some(token)) => snapshot(Some(g), kind::TOKEN, code::OK, Some(token)),
                    Ok(None) => snapshot(Some(g), kind::STOPPED, code::OK, None),
                    Err(error) => failure(Some(g), error),
                },
            },
            Command::Reset => {
                if let Some(g) = &mut generation {
                    g.reset();
                }
                cancel.store(false, Ordering::Release);
                snapshot(generation.as_ref(), kind::RESET, code::OK, None)
            }
            #[cfg(test)]
            Command::Panic => panic!("test worker panic containment"),
            #[cfg(test)]
            Command::Pause(entered, release) => {
                entered.wait();
                release.wait();
                snapshot(generation.as_ref(), kind::RESET, code::OK, None)
            }
        };
        if event.code == code::EXECUTION_FAILED {
            // No recovery or valid cache is exposed after execution failures.
            fatal.store(true, Ordering::Release);
            let _ = events.send(event);
            break;
        }
        if events.send(event).is_err() {
            break;
        }
    }
}
