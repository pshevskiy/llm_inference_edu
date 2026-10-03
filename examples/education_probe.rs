//! Opt-in CPU reference parity and factual checks, never authoritative grading.
use llm_inference::{
    education::{EXERCISES, PROBE_SYSTEM, check_generated_facts},
    generation::GenerationSession,
    model::Decoder,
    sampling::Sampling,
    tokenizer::{ChatTokenizer, Message, Role},
    weights::{Inventory, load_qwen},
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::Path,
    sync::atomic::AtomicBool,
};
#[derive(Deserialize)]
struct Case {
    id: String,
    prompt_ids: Vec<usize>,
    generated_ids: Vec<usize>,
    decoded: String,
    file: String,
    bytes: usize,
    sha256: String,
    fact_gate: bool,
    ended_eos: bool,
}
#[derive(Deserialize)]
struct Reference {
    status: String,
    repository: String,
    revision: String,
    weight_sha256: String,
    device: String,
    dtype: String,
    threads: usize,
    capacity: usize,
    max_new: usize,
    vocab_size: usize,
    atol: f64,
    rtol: f64,
    cases: Vec<Case>,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn save(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn host(program: &str, args: &[&str]) -> Option<String> {
    let r = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    r.status
        .success()
        .then(|| String::from_utf8_lossy(&r.stdout).trim().to_string())
}
fn run(owned: &mut Option<std::path::PathBuf>) -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("release build required".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: education_probe MODEL REFERENCE_MANIFEST FRESH_OUTPUT".into());
    }
    let manifest = Path::new(&args[1]);
    let raw = std::fs::read(manifest).map_err(|e| e.to_string())?;
    let r: Reference = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    let inventory: Inventory =
        serde_json::from_str(include_str!("../fixtures/qwen-inventory.json"))
            .map_err(|e| e.to_string())?;
    if r.status != "reference_complete"
        || r.repository != inventory.repository
        || r.revision != inventory.revision
        || r.weight_sha256 != inventory.assets["model.safetensors"].sha256
        || r.device != "cpu"
        || r.dtype != "FP32"
        || r.threads != 1
        || r.capacity != 512
        || r.max_new != 96
        || r.vocab_size != 151936
        || r.atol != 0.001
        || r.rtol != 0.0001
        || r.cases.len() != 3
    {
        return Err("invalid reference contract".into());
    }
    let model_dir = Path::new(&args[0]);
    let tokenizer = ChatTokenizer::load(model_dir, &inventory).map_err(|e| format!("{e:?}"))?;
    for (i, (case, e)) in r.cases.iter().zip(EXERCISES).enumerate() {
        let prompt = tokenizer
            .chat(
                &[
                    Message {
                        role: Role::System,
                        content: PROBE_SYSTEM,
                    },
                    Message {
                        role: Role::User,
                        content: &e.question(),
                    },
                ],
                true,
            )
            .map_err(|e| format!("{e:?}"))?;
        if case.id != e.id
            || case.prompt_ids != prompt
            || prompt.len() + 96 > 512
            || case.generated_ids.is_empty()
            || case.generated_ids.len() > 96
            || case.generated_ids.iter().any(|id| *id >= 151936)
            || case.file != format!("case{i}.bin")
            || case.bytes != case.generated_ids.len() * 151936 * 4
        {
            return Err("invalid case admission".into());
        }
        let path = manifest
            .parent()
            .ok_or("reference parent")?
            .join(&case.file);
        if std::fs::metadata(&path).map_err(|e| e.to_string())?.len() != case.bytes as u64 {
            return Err("reference length mismatch".into());
        }
        let mut input = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut h = Sha256::new();
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let n = input.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            h.update(&buffer[..n]);
        }
        if format!("{:x}", h.finalize()) != case.sha256 {
            return Err("reference hash mismatch".into());
        }
    }
    let output = Path::new(&args[2]);
    if output.exists() {
        return Err("fresh output required".into());
    }
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    *owned = Some(output.to_path_buf());
    let mut source = Sha256::new();
    let mut paths = std::fs::read_dir("src")
        .map_err(|e| e.to_string())?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    paths.extend([
        "Cargo.toml".into(),
        "Cargo.lock".into(),
        "examples/education_probe.rs".into(),
    ]);
    paths.sort();
    for p in paths {
        let name = p.to_string_lossy();
        let data = std::fs::read(&p).map_err(|e| e.to_string())?;
        source.update((name.len() as u64).to_le_bytes());
        source.update(name.as_bytes());
        source.update((data.len() as u64).to_le_bytes());
        source.update(data);
    }
    save(
        &output.join("identity.json"),
        &json!({"status":"started","source_sha256":format!("{:x}",source.finalize()),"executable_sha256":hash(&std::fs::read(std::env::current_exe().map_err(|e|e.to_string())?).map_err(|e|e.to_string())?),"reference_sha256":hash(&raw),"revision":r.revision,"atol":0.001,"rtol":0.0001,"capacity":512,"max_new":96,"host":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"version":host("sw_vers",&["-productVersion"]),"build":host("sw_vers",&["-buildVersion"]),"chip":host("sysctl",&["-n","machdep.cpu.brand_string"]),"memory_bytes":host("sysctl",&["-n","hw.memsize"]),"rustc":host("rustc",&["-vV"])}}),
    )?;
    let model = Decoder::from_qwen(
        load_qwen(model_dir, &inventory).map_err(|e| format!("{e:?}"))?,
        512,
    )
    .map_err(|e| format!("{e:?}"))?;
    let cancel = AtomicBool::new(false);
    let mut positions =
        std::fs::File::create(output.join("positions.jsonl")).map_err(|e| e.to_string())?;
    let mut records = Vec::new();
    for case in &r.cases {
        let mut generation =
            GenerationSession::new(&model, 512, Sampling::Greedy, tokenizer.stop_tokens(), 96)
                .map_err(|e| format!("{e:?}"))?;
        generation
            .start(&case.prompt_ids)
            .map_err(|e| format!("{e:?}"))?;
        let mut reference =
            std::fs::File::open(manifest.parent().ok_or("parent")?.join(&case.file))
                .map_err(|e| e.to_string())?;
        let mut bytes = vec![0; 151936 * 4];
        let mut generated = Vec::new();
        let mut budget = 0_f32;
        for &expected_id in &case.generated_ids {
            let id = generation
                .next_token(&cancel)
                .map_err(|e| format!("{e:?}"))?
                .ok_or("early stop")?;
            reference
                .read_exact(&mut bytes)
                .map_err(|e| e.to_string())?;
            let actual = generation.prefix_logits().ok_or("missing prefix logits")?;
            for (&a, word) in actual.iter().zip(bytes.chunks_exact(4)) {
                let b = f32::from_le_bytes(word.try_into().map_err(|_| "row")?);
                if !a.is_finite() || !b.is_finite() {
                    return Err("nonfinite logits".into());
                }
                budget = budget.max((a - b).abs() / (0.001 + 0.0001 * b.abs()));
            }
            writeln!(positions,"{}",json!({"case":case.id,"emission":generated.len(),"processed":generation.processed(),"history":generation.history().len(),"max_budget":budget,"greedy_equal":id==expected_id})).map_err(|e|e.to_string())?;
            positions.flush().map_err(|e| e.to_string())?;
            if id != expected_id || budget > 1.0 {
                return Err(format!(
                    "frozen numeric/greedy gate failed {} budget{budget}",
                    case.id
                ));
            }
            generated.push(id);
        }
        if generation
            .next_token(&cancel)
            .map_err(|e| format!("{e:?}"))?
            .is_some()
        {
            return Err("reference stopped early".into());
        }
        let text = tokenizer
            .decode(&generated, true)
            .map_err(|e| format!("{e:?}"))?;
        if text != case.decoded
            || generation.processed() + 1 != generation.history().len()
            || case.ended_eos
                != tokenizer
                    .stop_tokens()
                    .contains(generated.last().ok_or("empty")?)
        {
            return Err("text/cache invariant mismatch".into());
        }
        let factual = check_generated_facts(&case.id, &text).is_ok() && case.ended_eos;
        if factual != case.fact_gate {
            return Err("factual gate mismatch".into());
        }
        println!("{}: fact_gate={factual}; {text}", case.id);
        records.push(json!({"id":case.id,"text":text,"numeric_pass":true,"max_budget":budget,"fact_gate":factual,"ended_eos":case.ended_eos,"processed":generation.processed(),"history":generation.history().len(),"generated_tokens":generated.len()}));
        save(&output.join("progress.json"), &json!({"cases":records}))?;
    }
    save(
        &output.join("result.json"),
        &json!({"status":"passed_numerical","cases":records,"normal_generated_mode":"disabled: matching fields does not approve prose","curated_lessons":"available independently","teacher_quality":"pending independent prose review; no general reliability claim"}),
    )?;
    save(
        &output.join("completion.json"),
        &json!({"status":"passed_numerical"}),
    )?;
    Ok(())
}
fn main() {
    let mut output = None;
    if let Err(error) = run(&mut output) {
        if let Some(path) = output {
            let _ = save(
                &path.join("completion.json"),
                &json!({"status":"failed","error":error}),
            );
        }
        eprintln!("{error}");
        std::process::exit(1);
    }
}
