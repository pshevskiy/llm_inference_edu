//! Opt-in CPU context qualification. Stream reference rows; never allocate T×V
//! logits or an attention matrix. Performance samples run after numerical gates.
use llm_inference::benchmark::{distribution, scratch_estimate};
use llm_inference::generation::GenerationSession;
use llm_inference::kv_cache::CachedSession;
use llm_inference::model::Decoder;
use llm_inference::sampling::Sampling;
use llm_inference::weights::{Inventory, load_qwen};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
#[derive(Deserialize)]
struct RequestRow {
    name: String,
    file: String,
    elements: usize,
    sha256: String,
    greedy: usize,
}
#[derive(Deserialize)]
struct Request {
    prompt_ids: Vec<usize>,
    tool_fixture_ids: Vec<usize>,
    generated_ids: Vec<usize>,
    continuation_id: usize,
    rows: Vec<RequestRow>,
}
#[derive(Deserialize)]
struct Reference {
    status: String,
    request: Request,
    repository: String,
    revision: String,
    weight_sha256: String,
    capacity: usize,
    ids: Vec<usize>,
    vocab_size: usize,
    workload_sha256: String,
    logits_file: String,
    logits_bytes: u64,
    logits_sha256: String,
    logit_atol: f32,
    rtol: f32,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn file_hash(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut buffer = vec![0; 4 * 1024 * 1024];
    let mut digest = Sha256::new();
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn command(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}
fn rss() -> Option<u64> {
    command("ps", &["-o", "rss=", "-p", &std::process::id().to_string()])?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}
fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().fold(
        0,
        |best, (i, &value)| if value > values[best] { i } else { best },
    )
}
fn source_hash() -> Result<String, String> {
    let mut paths = std::fs::read_dir("src")
        .map_err(|e| e.to_string())?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    paths.extend([
        "Cargo.toml".into(),
        "Cargo.lock".into(),
        "examples/context_qualification.rs".into(),
    ]);
    paths.sort();
    let mut digest = Sha256::new();
    for path in paths {
        let name = path.to_string_lossy();
        let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(&bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn gate(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) -> Result<Value, String> {
    if actual.len() != expected.len() {
        return Err("logit length mismatch".into());
    }
    let mut max_abs = 0.0_f32;
    let mut max_budget = 0.0_f32;
    for (&a, &b) in actual.iter().zip(expected) {
        if !a.is_finite() || !b.is_finite() {
            return Err("nonfinite request logits".into());
        }
        let error = (a - b).abs();
        max_abs = max_abs.max(error);
        max_budget = max_budget.max(error / (atol + rtol * b.abs()));
    }
    let greedy_equal = argmax(actual) == argmax(expected);
    if max_budget > 1.0 || !greedy_equal {
        return Err(format!(
            "request frozen gate failed budget{max_budget},greedy{greedy_equal}"
        ));
    }
    Ok(json!({"max_abs":max_abs,"max_budget":max_budget,"greedy_equal":greedy_equal}))
}
fn run(owned_output: &mut Option<std::path::PathBuf>) -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("use a release build".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: context_qualification LOCAL_MODEL REFERENCE_MANIFEST NEW_RESULT_DIR".into(),
        );
    }
    let manifest_path = Path::new(&args[1]);
    let manifest_bytes = std::fs::read(manifest_path).map_err(|e| e.to_string())?;
    let reference: Reference =
        serde_json::from_slice(&manifest_bytes).map_err(|e| e.to_string())?;
    let inventory: Inventory =
        serde_json::from_str(include_str!("../fixtures/qwen-inventory.json")).unwrap();
    let capacity = reference.capacity;
    if reference.status != "passed"
        || ![32, 128, 256, 512, 1024, 2048].contains(&capacity)
        || reference.ids.len() != capacity
        || reference.vocab_size != 151936
        || reference.ids.iter().any(|&id| id >= reference.vocab_size)
        || reference.repository != inventory.repository
        || reference.revision != inventory.revision
        || reference.weight_sha256 != inventory.assets["model.safetensors"].sha256
        || reference.logit_atol != 0.001
        || reference.rtol != 0.0001
        || reference.logits_file != "logits.bin"
        || reference.logits_bytes != (capacity * reference.vocab_size * 4) as u64
        || reference.workload_sha256 != hash(&serde_json::to_vec(&reference.ids).unwrap())
    {
        return Err("invalid reference identity/layout/frozen gate".into());
    }
    let logits_path = manifest_path.parent().unwrap().join(&reference.logits_file);
    if std::fs::metadata(&logits_path)
        .map_err(|e| e.to_string())?
        .len()
        != reference.logits_bytes
        || file_hash(&logits_path)? != reference.logits_sha256
    {
        return Err("reference bytes mismatch".into());
    }
    if reference.request.prompt_ids != reference.ids[..capacity - 12]
        || reference.request.tool_fixture_ids != reference.ids[capacity - 6..capacity - 2]
        || reference.request.generated_ids.len() != 2
        || reference.request.rows.len() != 3
    {
        return Err("invalid request oracle layout".into());
    }
    let mut request_expected = Vec::new();
    for (row, name) in reference.request.rows.iter().zip([
        "request-prefill",
        "request-decode",
        "request-continuation",
    ]) {
        if row.name != name
            || row.file != format!("{name}.bin")
            || row.elements != reference.vocab_size
            || row.greedy >= reference.vocab_size
        {
            return Err("invalid request oracle row".into());
        }
        let payload = std::fs::read(manifest_path.parent().unwrap().join(&row.file))
            .map_err(|e| e.to_string())?;
        if payload.len() != row.elements * 4 || hash(&payload) != row.sha256 {
            return Err("request oracle hash/length mismatch".into());
        }
        let values: Vec<f32> = payload
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        if values.iter().any(|v| !v.is_finite()) || argmax(&values) != row.greedy {
            return Err("invalid oracle values".into());
        }
        request_expected.push(values);
    }
    if reference.request.generated_ids
        != [
            reference.request.rows[0].greedy,
            reference.request.rows[1].greedy,
        ]
        || reference.request.continuation_id != reference.request.rows[2].greedy
    {
        return Err("request oracle IDs inconsistent".into());
    }
    let output = Path::new(&args[2]);
    if output.exists() {
        return Err("choose a fresh result directory".into());
    }
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    *owned_output = Some(output.to_path_buf());
    let identity = json!({"status":"running","source_sha256":source_hash()?,
        "executable_sha256":file_hash(&std::env::current_exe().map_err(|e|e.to_string())?)?,
        "reference_manifest_sha256":hash(&manifest_bytes),"workload_sha256":reference.workload_sha256,
        "repository":reference.repository,"revision":reference.revision,"weight_sha256":reference.weight_sha256,
        "capacity":capacity,"compute_dtype":"FP32","threads":1,"logit_atol":reference.logit_atol,"rtol":reference.rtol,
        "host":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH,
        "chip":command("sysctl",&["-n","machdep.cpu.brand_string"]),"memory":command("sysctl",&["-n","hw.memsize"]),
        "os_version":command("sw_vers",&["-productVersion"]),"os_build":command("sw_vers",&["-buildVersion"]),
        "rustc":command("rustc",&["-vV"]),"pressure":command("memory_pressure",&["-Q"])}});
    std::fs::write(
        output.join("identity.json"),
        serde_json::to_vec_pretty(&identity).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    let start = Instant::now();
    let model = Decoder::from_qwen(
        load_qwen(Path::new(&args[0]), &inventory).map_err(|e| format!("{e:?}"))?,
        capacity,
    )
    .map_err(|e| format!("{e:?}"))?;
    let load_ns = start.elapsed().as_nanos();
    let weights = model.logical_weight_bytes().map_err(|e| format!("{e:?}"))?;
    let mut session = CachedSession::new(&model, capacity).map_err(|e| format!("{e:?}"))?;
    let scratch = scratch_estimate(model.config(), capacity)?;
    let admission = weights
        .checked_add(session.logical_bytes())
        .and_then(|n| n.checked_add(scratch))
        .and_then(|n| n.checked_add(64 * 1024 * 1024))
        .ok_or("budget overflow")?;
    if admission > 6 * 1024 * 1024 * 1024 {
        return Err("logical admission exceeds 6GiB".into());
    }
    let mut actual = vec![0.; reference.vocab_size];
    let mut bytes = vec![0; reference.vocab_size * 4];
    let mut expected = vec![0.; reference.vocab_size];
    let mut stream = std::fs::File::open(&logits_path).map_err(|e| e.to_string())?;
    let mut raw =
        std::fs::File::create(output.join("positions.jsonl")).map_err(|e| e.to_string())?;
    let mut max_abs = 0.0_f32;
    let mut max_budget = 0.0_f32;
    let mut peak_sampled_rss = rss();
    for (position, &token) in reference.ids.iter().enumerate() {
        stream.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        for (value, chunk) in expected.iter_mut().zip(bytes.chunks_exact(4)) {
            *value = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        session
            .step_into(token, &mut actual)
            .map_err(|e| format!("{e:?}"))?;
        let mut row_abs = 0.0_f32;
        let mut row_budget = 0.0_f32;
        for (&a, &b) in actual.iter().zip(&expected) {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite logits".into());
            }
            let error = (a - b).abs();
            row_abs = row_abs.max(error);
            row_budget = row_budget.max(error / (reference.logit_atol + reference.rtol * b.abs()));
        }
        let greedy_equal = argmax(&actual) == argmax(&expected);
        writeln!(raw,"{}",json!({"position":position,"processed":session.processed(),"max_abs":row_abs,"max_budget":row_budget,"greedy_equal":greedy_equal})).map_err(|e|e.to_string())?;
        max_abs = max_abs.max(row_abs);
        max_budget = max_budget.max(row_budget);
        if row_budget > 1.0 || !greedy_equal {
            return Err(format!(
                "frozen gate failed at position{position}: budget{row_budget}, greedy{greedy_equal}"
            ));
        }
        assert_eq!(session.processed(), position + 1);
        if (position + 1) % 32 == 0 {
            let observed = rss();
            peak_sampled_rss = peak_sampled_rss.into_iter().chain(observed).max();
            if observed.is_some_and(|n| n > 6 * 1024 * 1024 * 1024) {
                return Err("sampled RSS exceeds6GiB".into());
            }
            println!(
                "Rust {}/{capacity}, max_abs={max_abs:.8e}, max_budget={max_budget:.6}",
                position + 1
            );
            raw.flush().map_err(|e| e.to_string())?;
        }
    }
    drop(session); // No overlapping second KV allocation in benchmark scopes.
    let cancel = AtomicBool::new(false);
    let prompt = &reference.ids[..capacity - 12];
    let mut generation = GenerationSession::new(&model, capacity, Sampling::Greedy, &[], 2)
        .map_err(|e| format!("{e:?}"))?;
    let mut samples = Vec::new();
    // Long prefill is one bounded sample; no robust distribution claim. Decode
    // and continuation are included in this actual stateful request.
    let start = Instant::now();
    generation.start(prompt).map_err(|e| format!("{e:?}"))?;
    let prefill_ns = start.elapsed().as_nanos();
    let first = generation
        .next_token(&cancel)
        .map_err(|e| format!("{e:?}"))?
        .unwrap();
    let ttft_ns = start.elapsed().as_nanos();
    let first_gate = gate(
        generation.prefix_logits().ok_or("missing prefill logits")?,
        &request_expected[0],
        reference.logit_atol,
        reference.rtol,
    )?;
    if first != reference.request.generated_ids[0] {
        return Err("first ID mismatch".into());
    }
    let start = Instant::now();
    let second = generation
        .next_token(&cancel)
        .map_err(|e| format!("{e:?}"))?
        .unwrap();
    let decode_ns = start.elapsed().as_nanos();
    let second_gate = gate(
        generation.prefix_logits().ok_or("missing decode logits")?,
        &request_expected[1],
        reference.logit_atol,
        reference.rtol,
    )?;
    if second != reference.request.generated_ids[1] {
        return Err("second ID mismatch".into());
    }
    let before_processed = generation.processed();
    let before_history = generation.history().len();
    let tool_ids = &reference.ids[capacity - 6..capacity - 2];
    let start = Instant::now();
    generation
        .append_context(tool_ids, 2, &cancel)
        .map_err(|e| format!("{e:?}"))?;
    let after_processed = generation.processed();
    let next = generation
        .next_token(&cancel)
        .map_err(|e| format!("{e:?}"))?
        .unwrap();
    let continuation_ns = start.elapsed().as_nanos();
    let continuation_gate = gate(
        generation
            .prefix_logits()
            .ok_or("missing continuation logits")?,
        &request_expected[2],
        reference.logit_atol,
        reference.rtol,
    )?;
    if next != reference.request.continuation_id {
        return Err("continuation ID mismatch".into());
    }
    assert_eq!(before_history, before_processed + 1);
    assert_eq!(after_processed, before_history + tool_ids.len());
    assert_eq!(generation.processed(), after_processed);
    assert_eq!(generation.history().len(), after_processed + 1);
    samples.push(
        json!({"prefill_ns":prefill_ns,"ttft_ns":ttft_ns,"decode_ns":decode_ns,
        "decode_tokens_per_second":1e9/decode_ns as f64,"continuation_ns":continuation_ns,
        "prompt_tokens":prompt.len(),"generated_ids":[first,second],"tool_fixture_ids":tool_ids,
        "continuation_id":next,"before_processed":before_processed,"before_history":before_history,
        "after_processed":after_processed,"after_history":generation.history().len(),
        "kind":"one bounded sample; deterministic token suffix, no actual tool capability"}),
    );
    let (decode_stats, continuation_stats) = (
        distribution(&[decode_ns as f64 / 1e6])?,
        distribution(&[continuation_ns as f64 / 1e6])?,
    );
    let record = json!({"status":"passed","capacity":capacity,"positions_compared":capacity,
        "max_abs":max_abs,"max_budget":max_budget,"all_greedy_equal":true,
        "reference_manifest_sha256":hash(&manifest_bytes),"weight_sha256":reference.weight_sha256,
        "repository":reference.repository,"revision":reference.revision,"workload_sha256":reference.workload_sha256,
        "executable_sha256":identity["executable_sha256"],
        "load_ns":load_ns,"compute_dtype":"FP32","storage_dtype":"BF16","threads":1,
        "logical_weights_bytes":weights,"logical_kv_bytes":2*24*capacity*2*64*4,
        "scratch_estimate_bytes":scratch,"admission_with_allowance_bytes":admission,
        "peak_sampled_rss_bytes":peak_sampled_rss,"rss_after_continuation":rss(),
        "request_gates":[first_gate,second_gate,continuation_gate],"samples":samples,"decode_ms":decode_stats,"continuation_ms":continuation_stats,
        "host":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH,
        "os_version":command("sw_vers",&["-productVersion"]),"os_build":command("sw_vers",&["-buildVersion"]),
        "chip":command("sysctl",&["-n","machdep.cpu.brand_string"]),"memory_bytes":command("sysctl",&["-n","hw.memsize"]),
        "power":command("pmset",&["-g","batt"]).map(|s|s.lines().next().unwrap_or("").to_string()),
        "thermal":command("pmset",&["-g","therm"]),"pressure":command("memory_pressure",&["-Q"])},
        "energy":"unavailable; no energy instrument used","native_admission":"unchanged8",
        "precision_or_model_fallback":false});
    std::fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&record).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(output.join("completion.json"), b"{\"status\":\"passed\"}\n")
        .map_err(|e| e.to_string())?;
    println!(
        "Passed capacity{capacity}; {} positions; KV{}bytes",
        capacity,
        2 * 24 * capacity * 2 * 64 * 4
    );
    Ok(())
}
fn main() {
    let mut owned_output = None;
    if let Err(error) = run(&mut owned_output) {
        if let Some(output) = owned_output {
            let record = json!({"status":"failed","error":error});
            let _ = std::fs::write(
                output.join("completion.json"),
                serde_json::to_vec_pretty(&record).unwrap(),
            );
        }
        eprintln!("{error}");
        std::process::exit(1);
    }
}
