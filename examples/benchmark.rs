use llm_inference::benchmark::{
    Mode, Sample, Workload, distribution, run_sample, scratch_estimate,
};
use llm_inference::kv_cache::CachedSession;
use llm_inference::model::Decoder;
use llm_inference::tokenizer::ChatTokenizer;
use llm_inference::weights::{Inventory, load_qwen};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn command(program: &str, args: &[&str]) -> Option<String> {
    let result = Command::new(program).args(args).output().ok()?;
    result
        .status
        .success()
        .then(|| String::from_utf8_lossy(&result.stdout).trim().to_string())
}
fn host() -> Value {
    let hardware = command("system_profiler", &["-json", "SPHardwareDataType"])
        .and_then(|s| serde_json::from_str::<Value>(&s).ok());
    let info = hardware
        .as_ref()
        .and_then(|v| v["SPHardwareDataType"].get(0));
    // Allowlist avoids serial number, UUID and other private hardware fields.
    json!({"os":std::env::consts::OS,"arch":std::env::consts::ARCH,
        "model":info.and_then(|v|v.get("machine_model")),"chip":info.and_then(|v|v.get("chip_type")),"memory":info.and_then(|v|v.get("physical_memory")),
        "os_version":command("sw_vers",&["-productVersion"]),"os_build":command("sw_vers",&["-buildVersion"]),
        "power_source_snapshot":command("pmset",&["-g","batt"]).map(|s|s.lines().next().unwrap_or("").to_string()),
        "thermal_snapshot":command("pmset",&["-g","therm"]),
        "memory_pressure_snapshot":command("memory_pressure",&["-Q"]),
        "unavailable_field_policy":"null means unavailable, never an inferred measurement"})
}
fn rss_snapshot() -> Option<u64> {
    command("ps", &["-o", "rss=", "-p", &std::process::id().to_string()])?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}
fn source_hash() -> Result<String, String> {
    let mut paths = std::fs::read_dir("src")
        .map_err(|e| e.to_string())?
        .map(|p| p.map(|v| v.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    paths.extend([
        "Cargo.toml".into(),
        "Cargo.lock".into(),
        "examples/benchmark.rs".into(),
    ]);
    paths.sort();
    let mut hash = Sha256::new();
    for path in paths {
        let name = path.to_string_lossy();
        let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn prepare_output(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Err(
            "result directory already exists; choose a fresh path to preserve raw results".into(),
        );
    }
    std::fs::create_dir_all(path).map_err(|e| e.to_string())
}
fn save_metadata(output: &Path, metadata: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(metadata).map_err(|e| e.to_string())?;
    let temporary = output.join("metadata.json.tmp");
    std::fs::write(&temporary, bytes).map_err(|e| e.to_string())?;
    // Same-directory rename preserves the preceding identity on partial writes.
    std::fs::rename(temporary, output.join("metadata.json")).map_err(|e| e.to_string())
}
fn summarize(samples: &[Sample], mode: Mode) -> Result<Value, String> {
    let selected: Vec<_> = samples
        .iter()
        .filter(|s| std::mem::discriminant(&s.mode) == std::mem::discriminant(&mode))
        .collect();
    let values = |f: fn(&Sample) -> u128| {
        selected
            .iter()
            .map(|s| f(s) as f64 / 1e6)
            .collect::<Vec<_>>()
    };
    let ttft: Vec<_> = selected
        .iter()
        .filter_map(|s| s.ttft_ns.map(|n| n as f64 / 1e6))
        .collect();
    let decode: Vec<_> = selected
        .iter()
        .filter_map(|s| s.decode_tokens_per_second)
        .collect();
    Ok(
        json!({"mode":mode,"prefill_ms":distribution(&values(|s|s.prefill_ns))?,"ttft_ms":if ttft.is_empty(){None}else{Some(distribution(&ttft)?)},
        "continuation_ms":distribution(&values(|s|s.continuation_next_token_ns))?,"decode_tokens_per_second":if decode.is_empty(){None}else{Some(distribution(&decode)?)},
        "statistic":"median, min, max, median absolute deviation; no p95 claim"}),
    )
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 4 || args.len() > 5 || !matches!(args[0].as_str(), "tiny" | "qwen") {
        return Err("usage: benchmark tiny|qwen MODEL_OR_CHECKPOINT WORKLOAD_JSON NEW_RESULT_DIR [--single] (real repeated runs require explicit owner opt-in)".into());
    }
    let single = args.get(4).is_some_and(|s| s == "--single");
    if args.len() == 5 && !single {
        return Err("only --single is supported as a final option".into());
    }
    if cfg!(debug_assertions) {
        return Err("performance harness requires a release build".into());
    }
    let bytes = std::fs::read(&args[2]).map_err(|e| e.to_string())?;
    let mut workload: Workload = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if single {
        workload.warmups = 0;
        workload.repetitions = 1;
    }
    let output = Path::new(&args[3]);
    if output.exists() {
        return Err("result directory already exists".into());
    }
    let metadata_host = host();
    let source = source_hash()?;
    let (model, tokenizer, identity, model_file_bytes, load_ns) = if args[0] == "tiny" {
        let bytes = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
        let start = Instant::now();
        let model = Decoder::from_json(std::str::from_utf8(&bytes).map_err(|e| e.to_string())?)
            .map_err(|e| format!("{e:?}"))?;
        (
            model,
            None,
            json!({"kind":"synthetic","sha256":digest(&bytes),"storage_dtype":"JSON FP32 values"}),
            bytes.len() as u64,
            start.elapsed().as_nanos(),
        )
    } else {
        let directory = Path::new(&args[1]);
        let inventory: Inventory =
            serde_json::from_str(include_str!("../fixtures/qwen-inventory.json"))
                .map_err(|e| e.to_string())?;
        let start = Instant::now();
        let loaded = load_qwen(directory, &inventory).map_err(|e| format!("{e:?}"))?;
        let model = Decoder::from_qwen(loaded, 32).map_err(|e| format!("{e:?}"))?;
        let load_ns = start.elapsed().as_nanos();
        // Tokenizer is separate from model load timing and every backend scope.
        let tokenizer = ChatTokenizer::load(directory, &inventory).map_err(|e| format!("{e:?}"))?;
        let asset = &inventory.assets["model.safetensors"];
        (
            model,
            Some(tokenizer),
            json!({"repository":inventory.repository,"revision":inventory.revision,"sha256":asset.sha256,"storage_dtype":"BF16"}),
            asset.bytes,
            load_ns,
        )
    };
    workload.validate(model.config())?;
    let weights = model.logical_weight_bytes().map_err(|e| format!("{e:?}"))?;
    let kv = model
        .config()
        .layers
        .checked_mul(2)
        .and_then(|n| n.checked_mul(model.config().kv_heads))
        .and_then(|n| n.checked_mul(workload.capacity))
        .and_then(|n| n.checked_mul(model.config().head_dim))
        .and_then(|n| n.checked_mul(4))
        .ok_or("cache arithmetic overflow")?;
    let scratch = scratch_estimate(model.config(), workload.capacity)?;
    let tokenizer_allowance = if tokenizer.is_some() {
        64 * 1024 * 1024
    } else {
        0
    };
    let admission = weights
        .checked_add(kv)
        .and_then(|n| n.checked_add(scratch))
        .and_then(|n| n.checked_add(tokenizer_allowance))
        .and_then(|n| n.checked_add(1024 * 1024))
        .ok_or("admission arithmetic overflow")?;
    if admission > 6 * 1024 * 1024 * 1024 {
        return Err("estimated engine working set exceeds 6 GiB planning budget".into());
    }
    let mut session =
        CachedSession::new(&model, workload.capacity).map_err(|e| format!("{e:?}"))?;
    assert_eq!(session.logical_bytes(), kv);
    // Correctness anchor runs outside all benchmark scopes.
    let uncached = model
        .forward(&workload.prompt, false)
        .map_err(|e| format!("{e:?}"))?
        .logits;
    let cached = session
        .prefill(&workload.prompt)
        .map_err(|e| format!("{e:?}"))?;
    if cached != uncached {
        return Err("cached/uncached preflight parity failed".into());
    }
    drop(cached);
    drop(uncached);
    let rss_before = rss_snapshot();
    prepare_output(output)?;
    let mut ids_bytes = Vec::new();
    for &id in &workload.prompt {
        ids_bytes.extend_from_slice(&(id as u64).to_le_bytes());
    }
    let mut metadata = json!({"schema_version":1,"host":metadata_host,"compiler_runtime_snapshot":command("rustc",&["-Vv"]),"git_head":command("git",&["rev-parse","HEAD"]),
        "git_dirty":command("git",&["status","--porcelain"]).map(|s|!s.is_empty()),"working_tree_source_sha256":source,"executable_sha256":digest(&std::fs::read(std::env::current_exe().map_err(|e|e.to_string())?).map_err(|e|e.to_string())?),"provenance_scope":"source/compiler are runtime snapshots, not certified binary build provenance; documented commands rebuild first; executable hash identifies actual code","run_status":"running","cargo_lock_sha256":digest(&std::fs::read("Cargo.lock").map_err(|e|e.to_string())?),
        "model":identity,"model_config":model.config(),"compute_dtype":"FP32","quantization":"none","numerical_threads":1,
        "workload":workload,"workload_file_sha256":digest(&bytes),"prompt_ids_u64le_sha256":digest(&ids_bytes),"model_file_bytes":model_file_bytes,
        "model_load_ns":load_ns,"load_scope":if args[0]=="tiny"{"JSON parse/decoder validation; file read before timer"}else{"verified file read/hash/BF16 expansion/decoder validation; tokenizer excluded"},"filesystem_cache_state":"not controlled; one initial load, not a certified cold load","download_conversion_compilation":"separate preparation; not included or measured in this harness",
        "logical_weight_bytes":weights,"logical_cache_bytes":kv,"logical_workspace_allocated_bytes":session.workspace_bytes(),"logical_scratch_peak_estimate_bytes":scratch,"tokenizer_allowance_bytes":tokenizer_allowance,"admission_estimate_bytes":admission,
        "memory_estimate_scope":"FP32 payload + KV + resident reusable workspace plus up to two caller logits rows + reserved parser/metadata allowance; not resident memory",
        "rss_snapshot_before_bytes":rss_before,"rss_snapshot_after_bytes":null,"rss_scope":"point snapshots, not process peak; use documented /usr/bin/time -l separately",
        "energy":"unavailable; no instrument","ui_delivery":"not measured","first_visible_text_scope":"complete decoder chunk ready, no rendering/printing",
        "single_sample":single,"timestamp_utc":command("date",&["-u","+%Y-%m-%dT%H:%M:%SZ"])});
    save_metadata(output, &metadata)?;
    let mut raw = std::fs::File::create(output.join("raw.jsonl")).map_err(|e| e.to_string())?;
    let mut samples = Vec::with_capacity(workload.repetitions * 2);
    for mode in [Mode::TeacherForced, Mode::Greedy] {
        for _ in 0..workload.warmups {
            std::hint::black_box(run_sample(
                &mut session,
                &workload,
                mode,
                tokenizer.as_ref(),
            )?);
        }
        for repetition in 0..workload.repetitions {
            let sample = run_sample(&mut session, &workload, mode, tokenizer.as_ref())?;
            let row = json!({"repetition":repetition,"sample":sample});
            writeln!(
                raw,
                "{}",
                serde_json::to_string(&row).map_err(|e| e.to_string())?
            )
            .map_err(|e| e.to_string())?;
            samples.push(sample);
        }
    }
    let rss_after = rss_snapshot();
    metadata["rss_snapshot_after_bytes"] = json!(rss_after);
    metadata["run_status"] = json!("samples_complete_diagnostic_pending");
    save_metadata(output, &metadata)?;
    session.reset();
    session
        .prefill(&workload.prompt)
        .map_err(|e| format!("{e:?}"))?;
    let (_, profile) = session
        .step_profiled(workload.fixed_output[0])
        .map_err(|e| format!("{e:?}"))?;
    std::fs::write(
        output.join("diagnostic.json"),
        serde_json::to_string_pretty(&profile).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let summaries = [
        summarize(&samples, Mode::TeacherForced)?,
        summarize(&samples, Mode::Greedy)?,
    ];
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_string_pretty(&summaries).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut report = String::from(
        "# CPU baseline measurement\n\nSource/model/workload identity, host and memory fields are in metadata.json. Raw repetitions are in raw.jsonl; diagnostic timing is separate.\n\n| Mode | Prefill median ms | TTFT median ms | Decode median tokens/s | Continuation median ms |\n|---|---:|---:|---:|---:|\n",
    );
    for s in &summaries {
        report.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            s["mode"],
            s["prefill_ms"]["median"],
            s["ttft_ms"]["median"],
            s["decode_tokens_per_second"]["median"],
            s["continuation_ms"]["median"]
        ));
    }
    report.push_str("\nMedian/min/max/MAD only; ten samples do not establish robust p95. Single-sample mode has no distribution claim. Backend timing excludes tokenization, sampling, text and profile scopes. Decode formula is (N-1)/(last-first); EOS-respecting output length may differ. Continuation evaluates the pending emitted token before appended IDs. Teacher-forced TTFT is unavailable; first logits-ready is recorded as a separate proxy. RSS is point-in-time, not peak. Energy unavailable. Longer proposed workloads are not admitted by this initial 32-token model.\n");
    std::fs::write(output.join("REPORT.md"), report).map_err(|e| e.to_string())?;
    metadata["run_status"] = json!("complete");
    save_metadata(output, &metadata)?;
    println!(
        "Saved measurements to {}; repetitions={} per mode; no energy measurement",
        output.display(),
        workload.repetitions
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
