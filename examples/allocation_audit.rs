//! Development-only allocator instrumentation; never used for latency comparisons.
use llm_inference::kv_cache::CachedSession;
use llm_inference::model::Decoder;
use llm_inference::weights::{Inventory, load_qwen};
use serde_json::{Value, json};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
use std::path::Path;
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
fn measure<T>(operation: impl FnOnce() -> T) -> (T, Value) {
    let region = Region::new(GLOBAL);
    let result = operation();
    let stats = region.change();
    (
        result,
        json!({"allocations":stats.allocations,"deallocations":stats.deallocations,"reallocations":stats.reallocations,"bytes_allocated":stats.bytes_allocated,"bytes_deallocated":stats.bytes_deallocated,"bytes_reallocated":stats.bytes_reallocated}),
    )
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let model = if args.is_empty() {
        Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap()
    } else {
        assert_eq!(
            args.len(),
            1,
            "optional argument is the pinned Qwen directory"
        );
        let inventory: Inventory =
            serde_json::from_str(include_str!("../fixtures/qwen-inventory.json")).unwrap();
        Decoder::from_qwen(load_qwen(Path::new(&args[0]), &inventory).unwrap(), 32).unwrap()
    };
    let (mut session, construction) = measure(|| CachedSession::new(&model, 32).unwrap());
    session.step(1).unwrap();
    let (_, step) = measure(|| session.step(5).unwrap());
    session.reset();
    let (_, prefill) = measure(|| session.prefill(&[1, 5, 7]).unwrap());
    let (_, reset) = measure(|| session.reset());
    let mut logits = vec![0.0; model.config().vocab_size];
    session.reset();
    session.step_into(1, &mut logits).unwrap();
    let (_, step_into) = measure(|| session.step_into(5, &mut logits).unwrap());
    session.reset();
    let (_, prefill_into) = measure(|| session.prefill_into(&[1, 5, 7], &mut logits).unwrap());
    let mut generation = llm_inference::generation::GenerationSession::new(
        &model,
        32,
        llm_inference::sampling::Sampling::Greedy,
        &[],
        4,
    )
    .unwrap();
    generation.start(&[1, 5, 7]).unwrap();
    let cancel = std::sync::atomic::AtomicBool::new(false);
    generation.next_token(&cancel).unwrap();
    let (_, greedy_next) = measure(|| generation.next_token(&cancel).unwrap());
    println!("{}", serde_json::to_string_pretty(&json!({"scope":"Rust global System allocator requests; instrumentation excluded from latency harness; model load outside regions; returned output lives at snapshot","layers":model.config().layers,"vocab":model.config().vocab_size,"construction":construction,"legacy_step":step,"legacy_prefill3":prefill,"reset":reset,"buffered_step":step_into,"buffered_prefill3":prefill_into,"greedy_next_token":greedy_next,"workspace_bytes":session.workspace_bytes()})).unwrap());
}
