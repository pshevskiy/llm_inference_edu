//! Opt-in short checkpoint qualification, never an ordinary unit test.
use llm_inference::benchmark::{Mode, Sample, Workload, run_sample};
use llm_inference::kv_cache::CachedSession;
use llm_inference::model::Decoder;
use llm_inference::tokenizer::ChatTokenizer;
use llm_inference::weights::{Inventory, load_qwen};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Command;
use std::time::Instant;
fn command(p: &str, args: &[&str]) -> Option<String> {
    let r = Command::new(p).args(args).output().ok()?;
    r.status
        .success()
        .then(|| String::from_utf8_lossy(&r.stdout).trim().to_string())
}
fn rss() -> Option<u64> {
    command("ps", &["-o", "rss=", "-p", &std::process::id().to_string()])?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}
fn logits(model: &Decoder, ids: &[usize]) -> Vec<f32> {
    model.forward(ids, true).unwrap().trace.unwrap().logits
}
fn log_normalizer(row: &[f32]) -> f64 {
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    f64::from(maximum)
        + row
            .iter()
            .map(|&v| (f64::from(v) - f64::from(maximum)).exp())
            .sum::<f64>()
            .ln()
}
fn greedy(row: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best
}
fn compare(ids: &[usize], a: &[f32], b: &[f32], vocab: usize) -> Value {
    assert_eq!(a.len(), ids.len() * vocab);
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    let mut baseline_loss = 0.;
    let mut quantized_loss = 0.;
    let mut kl = 0.;
    let mut matches = 0usize;
    // The loss has a known next-token target for N-1 positions, while the
    // maximum logit gate covers every teacher-forced row, including the last.
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .fold(0.0_f64, f64::max);
    for position in 0..ids.len() - 1 {
        let x = &a[position * vocab..(position + 1) * vocab];
        let y = &b[position * vocab..(position + 1) * vocab];
        let ax = log_normalizer(x);
        let by = log_normalizer(y);
        let target = ids[position + 1];
        baseline_loss += ax - f64::from(x[target]);
        quantized_loss += by - f64::from(y[target]);
        matches += usize::from(greedy(x) == greedy(y));
        for (&x, &y) in x.iter().zip(y) {
            let log_p = f64::from(x) - ax;
            let log_q = f64::from(y) - by;
            kl += log_p.exp() * (log_p - log_q);
        }
    }
    let observations = ids.len() - 1;
    json!({"observations":observations,"baseline_loss_sum_nats":baseline_loss,"quantized_loss_sum_nats":quantized_loss,"kl_sum_nats":kl,"greedy_matches":matches,"max_absolute_logit_change":max_abs})
}
fn benchmark(model: &Decoder, workload: &Workload, tokenizer: &ChatTokenizer) -> Vec<Sample> {
    let mut session = CachedSession::new(model, 32).unwrap();
    [Mode::TeacherForced, Mode::Greedy]
        .into_iter()
        .map(|mode| run_sample(&mut session, workload, mode, Some(tokenizer)).unwrap())
        .collect()
}
fn run() -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("release build required".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: int8_qualification PINNED_QWEN_DIR NEW_RESULT_DIR (one short correctness/diagnostic run)".into());
    }
    let output = Path::new(&args[1]);
    if output.exists() {
        return Err("fresh result directory required".into());
    }
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let rule_bytes =
        std::fs::read("docs/experiments/16-quality-rule.json").map_err(|e| e.to_string())?;
    let rule: Value = serde_json::from_slice(&rule_bytes).map_err(|e| e.to_string())?;
    let group = rule["group_size"].as_u64().ok_or("invalid rule")? as usize;
    let digest = |b: &[u8]| format!("{:x}", Sha256::digest(b));
    let mut source = Vec::new();
    let mut paths = std::fs::read_dir("src")
        .map_err(|e| e.to_string())?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    paths.extend([
        "Cargo.toml".into(),
        "Cargo.lock".into(),
        "examples/int8_qualification.rs".into(),
    ]);
    paths.sort();
    for path in paths {
        let name = path.to_string_lossy();
        let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
        source.extend_from_slice(&(name.len() as u64).to_le_bytes());
        source.extend_from_slice(name.as_bytes());
        source.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        source.extend_from_slice(&bytes);
    }
    let inventory: Inventory =
        serde_json::from_str(include_str!("../fixtures/qwen-inventory.json")).unwrap();
    let directory = Path::new(&args[0]);
    let tokenizer = ChatTokenizer::load(directory, &inventory).unwrap();
    let texts: Vec<_> = rule["held_out_public_text"]
        .as_array()
        .ok_or("missing heldout")?
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    let ids: Vec<_> = texts.iter().map(|t| tokenizer.encode(t).unwrap()).collect();
    if ids.iter().any(|i| i.len() < 2 || i.len() > 32) {
        return Err("heldout outside admitted2..32tokens".into());
    }
    let hardware = command("system_profiler", &["-json", "SPHardwareDataType"])
        .and_then(|s| serde_json::from_str::<Value>(&s).ok());
    let info = hardware
        .as_ref()
        .and_then(|v| v["SPHardwareDataType"].get(0));
    let workload: Workload =
        serde_json::from_str(include_str!("../benchmarks/qwen-short.json")).unwrap();
    let mut workload = workload;
    workload.warmups = 0;
    workload.repetitions = 1;
    let identity = json!({"rule":rule,"rule_sha256":digest(&rule_bytes),"public_text":texts,"token_ids":ids,"model_revision":inventory.revision,"model_sha256":inventory.assets["model.safetensors"].sha256,"model_file_bytes":inventory.assets["model.safetensors"].bytes,"storage_source":"unchanged pinned BF16 file; expanded FP32 loader precedes conversion","compute":"FP32 activations/accumulation, INT8 weight-only candidate, FP32 norms/bias","group":group,"chip":info.and_then(|v|v.get("chip_type")),"host_memory":info.and_then(|v|v.get("physical_memory")),"host_model":info.and_then(|v|v.get("machine_model")),"os_build":command("sw_vers",&["-buildVersion"]),"power":command("pmset",&["-g","batt"]).map(|s|s.lines().next().unwrap_or("").to_string()),"compiler_runtime_snapshot":command("rustc",&["-Vv"]),"git_head":command("git",&["rev-parse","HEAD"]),"runtime_source_sha256":digest(&source),"executable_sha256":digest(&std::fs::read(std::env::current_exe().map_err(|e|e.to_string())?).map_err(|e|e.to_string())?),"provenance":"runtime source/compiler snapshots; actual binary hash; rebuild first","planning_budget_bytes":6usize*1024*1024*1024,"benchmark_workload":workload,"scope":"single warm requests per precision/mode, no dispersion/speedup/default-switch qualification; no energy or sustained thermal test"});
    std::fs::write(
        output.join("identity.json"),
        serde_json::to_vec_pretty(&identity).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    let start = Instant::now();
    let baseline = Decoder::from_qwen(load_qwen(directory, &inventory).unwrap(), 32).unwrap();
    let load_ns = start.elapsed().as_nanos();
    let baseline_bytes = baseline.logical_weight_bytes().unwrap();
    if baseline_bytes > 6 * 1024 * 1024 * 1024 {
        return Err("baseline budget exceeded".into());
    }
    let baseline_logits: Vec<_> = ids.iter().map(|ids| logits(&baseline, ids)).collect();
    let baseline_samples = benchmark(&baseline, &workload, &tokenizer);
    let baseline_rss = rss();
    let start = Instant::now();
    let quantized = baseline.into_int8(group).unwrap();
    let conversion_ns = start.elapsed().as_nanos();
    let quantized_bytes = quantized.logical_weight_bytes().unwrap();
    let post_conversion_rss = rss();
    let quantized_logits: Vec<_> = ids.iter().map(|ids| logits(&quantized, ids)).collect();
    let quantized_samples = benchmark(&quantized, &workload, &tokenizer);
    let quantized_rss = rss();
    let cases: Vec<_> = ids
        .iter()
        .zip(baseline_logits.iter().zip(&quantized_logits))
        .map(|(ids, (a, b))| compare(ids, a, b, quantized.config().vocab_size))
        .collect();
    let observations = cases
        .iter()
        .map(|c| c["observations"].as_u64().unwrap())
        .sum::<u64>() as f64;
    let sum = |field: &str| {
        cases
            .iter()
            .map(|c| c[field].as_f64().unwrap())
            .sum::<f64>()
    };
    let mean_loss_increase =
        (sum("quantized_loss_sum_nats") - sum("baseline_loss_sum_nats")) / observations;
    let mean_kl = sum("kl_sum_nats") / observations;
    let greedy_agreement = sum("greedy_matches") / observations;
    let max_logit = cases
        .iter()
        .map(|c| c["max_absolute_logit_change"].as_f64().unwrap())
        .fold(0., f64::max);
    let ratio = quantized_bytes as f64 / baseline_bytes as f64;
    let quality_pass = mean_loss_increase
        <= rule["maximum_mean_held_out_loss_increase_nats"]
            .as_f64()
            .unwrap()
        && mean_kl <= rule["maximum_mean_kl_fp32_to_int8_nats"].as_f64().unwrap()
        && greedy_agreement
            >= rule["minimum_teacher_forced_greedy_agreement"]
                .as_f64()
                .unwrap()
        && max_logit <= rule["maximum_absolute_logit_change"].as_f64().unwrap();
    let memory_pass = ratio
        <= rule["maximum_encoded_logical_weight_ratio"]
            .as_f64()
            .unwrap();
    let start = Instant::now();
    let reference = quantized.into_dequantized_reference().unwrap();
    let reference_expand_ns = start.elapsed().as_nanos();
    let mut exact_reference = true;
    for (ids, expected) in ids.iter().zip(&quantized_logits) {
        let actual = logits(&reference, ids);
        exact_reference &= actual == *expected;
    }
    let results = json!({"cases":cases,"observations":observations,"baseline_mean_loss_nats":sum("baseline_loss_sum_nats")/observations,"quantized_mean_loss_nats":sum("quantized_loss_sum_nats")/observations,"mean_loss_increase_nats":mean_loss_increase,"mean_kl_fp32_to_int8_nats":mean_kl,"greedy_agreement":greedy_agreement,"maximum_absolute_logit_change":max_logit,"baseline_logical_weight_bytes":baseline_bytes,"int8_logical_weight_bytes":quantized_bytes,"encoded_ratio":ratio,"rss_baseline_after_quality_bytes":baseline_rss,"rss_post_conversion_bytes":post_conversion_rss,"rss_quantized_after_quality_bytes":quantized_rss,"memory_scope":"point RSS, allocator/page retention may differ; wholeprocess peak includes FP32 loading; no encoded file export","load_ns":load_ns,"conversion_ns":conversion_ns,"reference_expansion_ns":reference_expand_ns,"baseline_single_samples":baseline_samples,"quantized_single_samples":quantized_samples,"exact_once_dequantized_reference":exact_reference,"quality_pass":quality_pass,"logical_memory_pass":memory_pass,"decision":if exact_reference&&quality_pass&&memory_pass{"retain explicit experimental INT8 API; FP32 default; performance unqualified"}else{"reject pretrained INT8 selection; preserve FP32 default and teaching experiment"},"default_precision":"FP32","repeated_model_benchmark":"not run; sustained opt-in pending","energy":"unavailable"});
    std::fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(&results).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(
        output.join("completion.json"),
        b"{\"run_status\":\"complete\"}",
    )
    .map_err(|e| e.to_string())?;
    println!(
        "quality_pass={quality_pass}, memory_pass={memory_pass}, exact_reference={exact_reference}; mean_loss_change={mean_loss_increase}, KL={mean_kl}, greedy={greedy_agreement}, maxlogit={max_logit}, encoded_ratio={ratio}"
    );
    if !exact_reference {
        return Err("fused candidate/reference mismatch".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
