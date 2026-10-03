//! Actual, opt-in bounded model→calculator→model qualification, separate from
//! numerical agreement. Requires a passed larger-context CPU result first.
use llm_inference::agent::{AgentSession, Phase, Reply};
use llm_inference::model::Decoder;
use llm_inference::tokenizer::{ChatTokenizer, Message, Role};
use llm_inference::weights::{Inventory, load_qwen};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn file_hash(path: &Path) -> Result<String, String> {
    let mut input = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut buffer = vec![0; 4 * 1024 * 1024];
    let mut digest = Sha256::new();
    loop {
        let n = input.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
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
fn ids(value: &Value) -> Result<Vec<usize>, String> {
    value
        .as_array()
        .ok_or("missing ID array")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or("invalid ID".into())
        })
        .collect()
}
fn gate(actual: &[f32], expected: &[f32]) -> Result<(f32, f32), String> {
    if actual.len() != expected.len() {
        return Err("row length mismatch".into());
    }
    let mut peak_abs = 0.0_f32;
    let mut peak_budget = 0.0_f32;
    for (&a, &b) in actual.iter().zip(expected) {
        if !a.is_finite() || !b.is_finite() {
            return Err("nonfinite logits".into());
        }
        let error = (a - b).abs();
        peak_abs = peak_abs.max(error);
        peak_budget = peak_budget.max(error / (0.001 + 0.0001 * b.abs()));
    }
    if peak_budget > 1.0 {
        return Err(format!("frozen logit gate failed: {peak_budget}"));
    }
    Ok((peak_abs, peak_budget))
}
fn command(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}
fn run(owned: &mut Option<std::path::PathBuf>) -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("use a release build".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: agent_qwen_qualification LOCAL_MODEL PASSED_CONTEXT_RESULT AGENT_REFERENCE_MANIFEST NEW_RESULT_DIR".into());
    }
    let proof_path = Path::new(&args[1]);
    let proof_bytes = std::fs::read(proof_path).map_err(|e| e.to_string())?;
    let proof: Value = serde_json::from_slice(&proof_bytes).map_err(|e| e.to_string())?;
    let identity: Value = serde_json::from_slice(
        &std::fs::read(proof_path.parent().unwrap().join("identity.json"))
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let capacity = proof["capacity"]
        .as_u64()
        .ok_or("missing earned capacity")? as usize;
    if proof["status"] != "passed"
        || !(256..=2048).contains(&capacity)
        || identity["source_sha256"] != source_hash()?
    {
        return Err("passed current-source CPU context proof required".into());
    }
    let path = Path::new(&args[2]);
    let manifest_bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes).map_err(|e| e.to_string())?;
    let inventory: Inventory =
        serde_json::from_str(include_str!("../fixtures/qwen-inventory.json")).unwrap();
    if manifest["status"] != "reference_complete"
        || manifest["revision"] != inventory.revision
        || proof["revision"] != inventory.revision
        || manifest["capacity"] != proof["capacity"]
        || manifest["qualification_sha256"] != hash(&proof_bytes)
        || manifest["max_new"] != 64
        || manifest["max_calls"] != 1
        || manifest["logit_atol"] != 0.001
        || manifest["rtol"] != 0.0001
    {
        return Err("invalid agent reference identity/gate".into());
    }
    let output = Path::new(&args[3]);
    if output.exists() {
        return Err("choose a fresh result path".into());
    }
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    *owned = Some(output.to_path_buf());
    std::fs::write(output.join("identity.json"),serde_json::to_vec_pretty(&json!({"status":"running",
        "context_result_sha256":hash(&proof_bytes),"reference_sha256":hash(&manifest_bytes),
        "source_sha256":source_hash()?,"executable_sha256":file_hash(&std::env::current_exe().map_err(|e|e.to_string())?)?,
        "probe_source_sha256":hash(&std::fs::read("examples/agent_qwen_qualification.rs").map_err(|e|e.to_string())?),
        "capacity":capacity,"revision":inventory.revision,"policy":"RustCPU_FP32,noGPU","calculator_permission":true,
        "host":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH,
        "chip":command("sysctl",&["-n","machdep.cpu.brand_string"]),"memory_bytes":command("sysctl",&["-n","hw.memsize"]),
        "os_version":command("sw_vers",&["-productVersion"]),"os_build":command("sw_vers",&["-buildVersion"]),
        "rustc":command("rustc",&["-vV"]),"pressure":command("memory_pressure",&["-Q"]) }})).unwrap()).map_err(|e|e.to_string())?;
    let tokenizer =
        ChatTokenizer::load(Path::new(&args[0]), &inventory).map_err(|e| format!("{e:?}"))?;
    let model = Decoder::from_qwen(
        load_qwen(Path::new(&args[0]), &inventory).map_err(|e| format!("{e:?}"))?,
        capacity,
    )
    .map_err(|e| format!("{e:?}"))?;
    let cancel = AtomicBool::new(false);
    let mut results = Vec::new();
    let cases = manifest["cases"].as_array().ok_or("missing cases")?;
    if cases.len() != 3 {
        return Err("expected3fixedcases".into());
    }
    for (case_index, case) in cases.iter().enumerate() {
        let messages = case["messages"]
            .as_array()
            .ok_or("missing messages")?
            .iter()
            .map(|message| {
                Ok(Message {
                    role: match message["role"].as_str() {
                        Some("system") => Role::System,
                        Some("user") => Role::User,
                        _ => return Err("unsupported probe role".to_string()),
                    },
                    content: message["content"]
                        .as_str()
                        .ok_or("missing message content")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let prompt = tokenizer
            .chat(&messages, true)
            .map_err(|e| format!("{e:?}"))?;
        if prompt != ids(&case["prompt_ids"])? {
            return Err("independent prompt tokenization mismatch".into());
        }
        let mut agent =
            AgentSession::new_with_stop_tokens(&model, capacity, 64, 1, tokenizer.stop_tokens())
                .map_err(|e| format!("{e:?}"))?;
        agent
            .start(&prompt, &cancel)
            .map_err(|e| format!("{e:?}"))?;
        let mut quality = false;
        let mut reason = "incomplete protocol".to_string();
        let mut observed_turns = Vec::new();
        let mut peak_abs = 0.0_f32;
        let mut peak_budget = 0.0_f32;
        for (turn_index, turn) in case["turns"]
            .as_array()
            .ok_or("missing turns")?
            .iter()
            .enumerate()
        {
            if turn_index > 1 {
                return Err("tool step bound exceeded in reference".into());
            }
            let generated = ids(&turn["generated_ids"])?;
            if generated.is_empty() || generated.len() > 64 {
                return Err("invalid output bound".into());
            }
            let name = format!("case{case_index}-turn{turn_index}.bin");
            let row_path = path.parent().unwrap().join(&name);
            if turn["file"] != name
                || turn["bytes"] != (generated.len() * 151936 * 4) as u64
                || std::fs::metadata(&row_path)
                    .map_err(|e| e.to_string())?
                    .len()
                    != (generated.len() * 151936 * 4) as u64
                || turn["sha256"] != file_hash(&row_path)?
            {
                return Err("agent oracle row identity mismatch".into());
            }
            let mut input = std::fs::File::open(row_path).map_err(|e| e.to_string())?;
            let mut bytes = vec![0; 151936 * 4];
            let mut expected = vec![0.; 151936];
            let mut observed = Vec::new();
            let mut step_ns = Vec::new();
            for expected_id in generated {
                input.read_exact(&mut bytes).map_err(|e| e.to_string())?;
                for (v, chunk) in expected.iter_mut().zip(bytes.chunks_exact(4)) {
                    *v = f32::from_le_bytes(chunk.try_into().unwrap());
                }
                let start = Instant::now();
                let id = agent.next_token(&cancel).map_err(|e| format!("{e:?}"))?;
                step_ns.push(start.elapsed().as_nanos());
                let (a, b) = gate(
                    agent.prefix_logits().ok_or("missing evaluated logits")?,
                    &expected,
                )?;
                peak_abs = peak_abs.max(a);
                peak_budget = peak_budget.max(b);
                if id != expected_id {
                    return Err("independent greedy ID mismatch".into());
                }
                observed.push(id);
            }
            if agent.phase() != Phase::AwaitingReply
                || agent.processed()
                    != turn["processed"].as_u64().ok_or("missing processed")? as usize
                || agent.history().len()
                    != turn["history"].as_u64().ok_or("missing history")? as usize
            {
                return Err("exact output/prefix phase mismatch".into());
            }
            let decoded = tokenizer
                .decode(&observed, true)
                .map_err(|e| format!("{e:?}"))?;
            if decoded != turn["decoded"].as_str().ok_or("missing decode")? {
                return Err("independent text mismatch".into());
            }
            observed_turns.push(json!({"text":decoded,"ids":observed,"processed":agent.processed(),"history":agent.history().len(),"step_ns":step_ns}));
            let parsed = serde_json::from_str::<Reply>(&decoded);
            if turn_index == 1 {
                match parsed {
                    Ok(Reply::Final { text })
                        if text == case["expected_answer"].as_str().ok_or("missing answer")? =>
                    {
                        agent
                            .prepare_reply(&decoded, true)
                            .map_err(|e| format!("{e:?}"))?;
                        quality = true;
                        reason = "passed".into();
                    }
                    _ => reason = "final answer/schema mismatch".into(),
                }
                break;
            }
            let correct_call = match &parsed {
                Ok(Reply::Tool { arguments, .. }) => {
                    serde_json::to_value(arguments.operation).unwrap() == case["expected_operation"]
                        && Some(arguments.left) == case["expected_left"].as_f64()
                        && Some(arguments.right) == case["expected_right"].as_f64()
                }
                _ => false,
            };
            if !correct_call {
                reason = "expected typed calculator request missing/mismatched".into();
                break;
            }
            let result = agent
                .prepare_reply(&decoded, true)
                .map_err(|e| format!("{e:?}"))?
                .ok_or("missing trusted result")?
                .to_string();
            if turn["result_text"] != result {
                return Err("independent arithmetic result mismatch".into());
            }
            let prefix = if tokenizer
                .stop_tokens()
                .contains(agent.history().last().unwrap())
            {
                "\n"
            } else {
                "<|im_end|>\n"
            };
            let framed = format!(
                "{prefix}<|im_start|>user\nCalculator data: {result}<|im_end|>\n<|im_start|>assistant\n"
            );
            let suffix = tokenizer.encode(&framed).map_err(|e| format!("{e:?}"))?;
            if turn["suffix_text"] != framed || ids(&turn["suffix_ids"])? != suffix {
                return Err("independent result framing mismatch".into());
            }
            let before_processed = agent.processed();
            let before_history = agent.history().len();
            let start = Instant::now();
            agent
                .append_result(&suffix, &cancel)
                .map_err(|e| format!("{e:?}"))?;
            observed_turns.last_mut().unwrap()["continuation"] = json!({"append_ns":start.elapsed().as_nanos(),
                "before_processed":before_processed,"before_history":before_history,"suffix_ids":suffix,
                "after_processed":agent.processed(),"after_history":agent.history().len()});
            if agent.processed() != before_history + suffix.len()
                || agent.history().len() != agent.processed()
            {
                return Err("pending EOS/result invariant failed".into());
            }
        }
        println!("case{case_index}: quality{quality}, {reason}, max_budget{peak_budget:.6}");
        results.push(json!({"quality_pass":quality,"reason":reason,"max_abs":peak_abs,"max_budget":peak_budget,
            "numerical_gate_pass":true,"turns":observed_turns,"transcript":agent.transcript(),"reference_quality_pass":case["quality_pass"]}));
        std::fs::write(
            output.join("progress.json"),
            serde_json::to_vec_pretty(&results).unwrap(),
        )
        .map_err(|e| e.to_string())?;
    }
    let passed = results.iter().filter(|c| c["quality_pass"] == true).count();
    let record = json!({"status":"qualification_complete","numerical_gate":"passed","quality_passed":passed,
        "cases":results,"cases_total":3,"capacity":capacity,"reliability_claim":"three fixed probes only; no general agent capability",
        "private_prompts":"none; explicit public fixed arithmetic probes"});
    std::fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&record).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!("Capability: {passed}/3; numerical gates all passed");
    Ok(())
}
fn main() {
    let mut owned = None;
    if let Err(error) = run(&mut owned) {
        if let Some(path) = owned {
            let _ = std::fs::write(
                path.join("failure.json"),
                serde_json::to_vec_pretty(&json!({"status":"failed","error":error})).unwrap(),
            );
        }
        eprintln!("{error}");
        std::process::exit(1);
    }
}
