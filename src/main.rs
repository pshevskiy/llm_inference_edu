use llm_inference::generation::{GenerationSession, uncached_greedy};
use llm_inference::model::Decoder;
use llm_inference::sampling::Sampling;
use llm_inference::tokenizer::{ChatTokenizer, Message, Role};
use llm_inference::weights::{Inventory, load_qwen};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=5).contains(&args.len()) {
        return Err(
            "usage: llm_inference CHECKPOINT_DIR PROMPT [MAX_NEW=8] [SYSTEM=Be brief.] [MODE=cached|uncached]".into(),
        );
    }
    let max_new = args
        .get(2)
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|_| "MAX_NEW must be a positive integer")?
        .unwrap_or(8);
    let system = args.get(3).map(String::as_str).unwrap_or("Be brief.");
    let mode = args.get(4).map(String::as_str).unwrap_or("cached");
    if !matches!(mode, "cached" | "uncached") {
        return Err("MODE must be cached or uncached".into());
    }
    let inventory: Inventory =
        serde_json::from_str(include_str!("../fixtures/qwen-inventory.json"))
            .map_err(|e| e.to_string())?;
    let directory = Path::new(&args[0]);
    let load_start = Instant::now();
    let tokenizer =
        ChatTokenizer::load(directory, &inventory).map_err(|e| format!("tokenizer: {e:?}"))?;
    let prompt = tokenizer
        .chat(
            &[
                Message {
                    role: Role::System,
                    content: system,
                },
                Message {
                    role: Role::User,
                    content: &args[1],
                },
            ],
            true,
        )
        .map_err(|e| format!("template: {e:?}"))?;
    // Admit before expensive weight loading. This baseline's explicit cap is 32.
    if max_new == 0 || prompt.len().checked_add(max_new).is_none_or(|n| n > 32) {
        return Err(format!(
            "prompt {} + requested output {max_new} must fit the 32-token capacity; choose a shorter prompt/system or output limit",
            prompt.len()
        ));
    }
    let model = Decoder::from_qwen(
        load_qwen(directory, &inventory).map_err(|e| format!("load: {e:?}"))?,
        32,
    )
    .map_err(|e| format!("model: {e:?}"))?;
    let load_time = load_start.elapsed();
    let generation_start = Instant::now();
    let (emitted, stop) = if mode == "uncached" {
        let completion = uncached_greedy(&model, &prompt, tokenizer.stop_tokens(), max_new)
            .map_err(|e| format!("generation: {e:?}"))?;
        let text = tokenizer
            .decode(&completion.generated_ids, true)
            .map_err(|e| format!("decode: {e:?}"))?;
        print!("{text}");
        (completion.generated_ids.len(), Some(completion.stop))
    } else {
        let mut state = GenerationSession::new(
            &model,
            32,
            Sampling::Greedy,
            tokenizer.stop_tokens(),
            max_new,
        )
        .map_err(|e| format!("state: {e:?}"))?;
        state
            .start(&prompt)
            .map_err(|e| format!("prefill: {e:?}"))?;
        let cancel = AtomicBool::new(false);
        let mut stream = tokenizer.text_stream();
        while let Some(id) = state
            .next_token(&cancel)
            .map_err(|e| format!("generation: {e:?}"))?
        {
            if let Some(text) = stream.push(id).map_err(|e| format!("decode: {e:?}"))? {
                print!("{text}");
                io::stdout().flush().map_err(|e| e.to_string())?;
            }
        }
        if stream.pending_tokens() > 0 {
            eprintln!(
                "{} decoder tokens pending complete text; original IDs retained in history",
                stream.pending_tokens()
            );
        }
        eprintln!(
            "history={} evaluated={} (final emitted token may be pending)",
            state.history().len(),
            state.processed()
        );
        (state.generated(), state.stop_reason())
    };
    println!();
    eprintln!(
        "{mode} CPU FP32; prompt={} emitted={emitted} stop={stop:?}; load={:.3}s generation={:.3}s",
        prompt.len(),
        load_time.as_secs_f64(),
        generation_start.elapsed().as_secs_f64()
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
