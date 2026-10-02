//! Short fixed-shape CPU experiment; does not establish sustained model throughput.
use llm_inference::benchmark::distribution;
use llm_inference::{LinearError, kernels::linear_lanes4, linear};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;
type Kernel = fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<(), LinearError>;
fn command(program: &str, args: &[&str]) -> Option<String> {
    let r = Command::new(program).args(args).output().ok()?;
    r.status
        .success()
        .then(|| String::from_utf8_lossy(&r.stdout).trim().to_string())
}
fn run() -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("release build required".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("usage: kernel_benchmark NEW_RESULT_DIR".into());
    }
    let path = Path::new(&args[0]);
    if path.exists() {
        return Err("use a fresh result directory".into());
    }
    std::fs::create_dir_all(path).map_err(|e| e.to_string())?;
    let digest = |b: &[u8]| format!("{:x}", Sha256::digest(b));
    let mut source = Vec::new();
    for name in [
        "src/lib.rs",
        "src/kernels.rs",
        "examples/kernel_benchmark.rs",
        "Cargo.toml",
        "Cargo.lock",
    ] {
        source.extend_from_slice(name.as_bytes());
        source.extend_from_slice(&std::fs::read(name).map_err(|e| e.to_string())?);
    }
    let hardware = command("system_profiler", &["-json", "SPHardwareDataType"])
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok());
    let info = hardware
        .as_ref()
        .and_then(|v| v["SPHardwareDataType"].get(0));
    let mut metadata = json!({"run_status":"running","arch":std::env::consts::ARCH,"os":std::env::consts::OS,"os_version":command("sw_vers",&["-productVersion"]),"hardware_scope":"allowlisted actual host fields; null means unavailable","hardware_model":info.and_then(|v|v.get("machine_model")),"chip":info.and_then(|v|v.get("chip_type")),"memory":info.and_then(|v|v.get("physical_memory")),"power_snapshot":command("pmset",&["-g","batt"]).map(|s|s.lines().next().unwrap_or("").to_string()),"compiler_runtime_snapshot":command("rustc",&["-Vv"]),"git_head":command("git",&["rev-parse","HEAD"]),"working_tree_source_sha256":digest(&source),"executable_sha256":digest(&std::fs::read(std::env::current_exe().map_err(|e|e.to_string())?).map_err(|e|e.to_string())?),"source_scope":"runtime working files, not certified binary build provenance; rebuild immediately before running","input_recipe":"x[i]=((i*17%101)-50)/53 in FP32; W[j]=((j%251)-125)/257 in FP32","workload_kind":"synthetic public values with actual Qwen rectangular dimensions; one token; dtype FP32; threads1","warmups":3,"repetitions":10,"ordering":"alternate scalar/lane first per repetition","quality_gate":"finite and independent FP64 dot atol1e-5 + rtol1e-4; frozen operator gate","cache_state":"warm allocations, no explicit cache flush; weights may exceed CPU cache","energy":"unavailable","physical_memory":"not measured; checked payload <=6GiB","scope":"kernel only; input construction/reference/serialization outside timer; brief experiment, no sustained model or thermal claim"});
    std::fs::write(
        path.join("identity.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    let mut raw = std::fs::File::create(path.join("raw.jsonl")).map_err(|e| e.to_string())?;
    let mut summaries = Vec::new();
    for (name, input, output) in [
        ("vocabulary", 896usize, 151936usize),
        ("gate", 896, 4864),
        ("down", 4864, 896),
        ("attention_output", 896, 896),
        ("key", 896, 128),
    ] {
        let elements = input.checked_mul(output).ok_or("dimension overflow")?;
        let payload = elements
            .checked_add(input)
            .and_then(|n| output.checked_mul(4).and_then(|v| n.checked_add(v)))
            .and_then(|n| n.checked_mul(4))
            .ok_or("memory overflow")?;
        if payload > 6 * 1024 * 1024 * 1024 {
            return Err("shape exceeds budget".into());
        }
        let x: Vec<f32> = (0..input)
            .map(|i| ((i * 17 % 101) as f32 - 50.) / 53.)
            .collect();
        let w: Vec<f32> = (0..elements)
            .map(|j| ((j % 251) as f32 - 125.) / 257.)
            .collect();
        let mut ys = vec![0.; output];
        let mut yl = vec![0.; output];
        // Independent expected sums: accumulation is FP64, not either FP32 kernel.
        let expected: Vec<f64> = w
            .chunks_exact(input)
            .map(|row| {
                row.iter()
                    .zip(&x)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum()
            })
            .collect();
        linear(&x, &w, &mut ys, 1, input, output).unwrap();
        linear_lanes4(&x, &w, &mut yl, 1, input, output).unwrap();
        let mut max_error = [0.0_f64; 2];
        for (kind, actual) in [&ys, &yl].iter().enumerate() {
            for (&a, &e) in actual.iter().zip(&expected) {
                let error = (f64::from(a) - e).abs();
                max_error[kind] = max_error[kind].max(error);
                if !a.is_finite() || error > 1e-5 + 1e-4 * e.abs() {
                    return Err(format!("{name}: quality gate failed for kernel {kind}"));
                }
            }
        }
        for _ in 0..3 {
            linear(
                std::hint::black_box(&x),
                std::hint::black_box(&w),
                &mut ys,
                1,
                input,
                output,
            )
            .unwrap();
            linear_lanes4(
                std::hint::black_box(&x),
                std::hint::black_box(&w),
                &mut yl,
                1,
                input,
                output,
            )
            .unwrap();
        }
        let mut times = [Vec::new(), Vec::new()];
        for repetition in 0..10 {
            let order = if repetition % 2 == 0 { [0, 1] } else { [1, 0] };
            for kind in order {
                let kernel: Kernel = if kind == 0 { linear } else { linear_lanes4 };
                let y = if kind == 0 { &mut ys } else { &mut yl };
                let start = Instant::now();
                kernel(
                    std::hint::black_box(&x),
                    std::hint::black_box(&w),
                    y,
                    1,
                    input,
                    output,
                )
                .unwrap();
                let elapsed = start.elapsed().as_nanos();
                std::hint::black_box(y);
                times[kind].push(elapsed as f64 / 1e6);
                writeln!(raw,"{}",json!({"shape":name,"input_dim":input,"output_dim":output,"repetition":repetition,"kernel":if kind==0{"scalar"}else{"lanes4"},"elapsed_ns":elapsed})).map_err(|e|e.to_string())?;
            }
        }
        summaries.push(json!({"shape":name,"input_dim":input,"output_dim":output,"weight_payload_bytes":elements*4,"scalar_ms":distribution(&times[0])?,"lanes4_ms":distribution(&times[1])?,"max_abs_error_fp64":max_error}));
    }
    std::fs::write(
        path.join("summary.json"),
        serde_json::to_vec_pretty(&summaries).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    metadata["run_status"] = json!("complete");
    // Immutable identity is preserved; mutable completion is a separate artifact.
    std::fs::write(
        path.join("completion.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "Saved five rectangular shape experiments to {}",
        path.display()
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
