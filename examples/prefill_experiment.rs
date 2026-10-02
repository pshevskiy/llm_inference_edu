//! Projection experiment: matrix-vector versus token chunks, not full cached prefill.
use llm_inference::benchmark::distribution;
#[path = "support/prefill_kernel.rs"]
mod prefill_kernel;
use llm_inference::kernels::linear_lanes4;
use prefill_kernel::linear_chunked;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;
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
        return Err("usage: prefill_experiment NEW_RESULT_DIR".into());
    }
    let path = Path::new(&args[0]);
    if path.exists() {
        return Err("fresh result directory required".into());
    }
    std::fs::create_dir_all(path).map_err(|e| e.to_string())?;
    let hardware = command("system_profiler", &["-json", "SPHardwareDataType"])
        .and_then(|text| serde_json::from_str::<Value>(&text).ok());
    let info = hardware
        .as_ref()
        .and_then(|v| v["SPHardwareDataType"].get(0));
    let mut source = Vec::new();
    for name in [
        "src/lib.rs",
        "src/kernels.rs",
        "examples/support/prefill_kernel.rs",
        "examples/prefill_experiment.rs",
        "Cargo.toml",
        "Cargo.lock",
    ] {
        source.extend_from_slice(name.as_bytes());
        source.extend_from_slice(&std::fs::read(name).map_err(|e| e.to_string())?);
    }
    let digest = |b: &[u8]| format!("{:x}", Sha256::digest(b));
    let recipe = json!({"tokens":[1,4,8,16,32],"chunks":[1,2,4,8],"shapes":[[896,4864],[4864,896],[896,128]],"x":"((i*17%101)-50)/53 FP32","w":"((j%251)-125)/257 FP32"});
    let identity = json!({"recipe":recipe,"workload_sha256":digest(&serde_json::to_vec(&recipe).unwrap()),"chip":info.and_then(|v|v.get("chip_type")),"model":info.and_then(|v|v.get("machine_model")),"host_memory":info.and_then(|v|v.get("physical_memory")),"os_build":command("sw_vers",&["-buildVersion"]),"arch":std::env::consts::ARCH,"power":command("pmset",&["-g","batt"]).map(|s|s.lines().next().unwrap_or("").to_string()),"compiler_runtime_snapshot":command("rustc",&["-Vv"]),"git_head":command("git",&["rev-parse","HEAD"]),"working_tree_source_sha256":digest(&source),"executable_sha256":digest(&std::fs::read(std::env::current_exe().map_err(|e|e.to_string())?).map_err(|e|e.to_string())?),"provenance":"runtime source/compiler snapshots; rebuild first; executable hash identifies actual binary","warmups":3,"repetitions":10,"ordering":"rotating five variants each repetition","threads":1,"dtype":"FP32","scope":"projection only, synthetic public values/actual Qwen rectangular shapes; does not implement matrix prefill in decoder or KV state","cache":"warm allocations; no explicit flush","memory_scope":"checked logical payload, physical peak not measured","energy":"unavailable","blas":"not used","sustained":"not measured"});
    std::fs::write(
        path.join("identity.json"),
        serde_json::to_vec_pretty(&identity).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    let mut raw = std::fs::File::create(path.join("raw.jsonl")).map_err(|e| e.to_string())?;
    let mut summary = Vec::new();
    for (input, output) in [(896usize, 4864usize), (4864, 896), (896, 128)] {
        let weights = input.checked_mul(output).ok_or("overflow")?;
        let maximum_payload = 32usize
            .checked_mul(input)
            .and_then(|n| {
                32usize
                    .checked_mul(output)
                    .and_then(|m| m.checked_mul(7))
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| n.checked_add(weights))
            .and_then(|n| n.checked_mul(4))
            .ok_or("overflow")?;
        if maximum_payload > 6 * 1024 * 1024 * 1024 {
            return Err("budget exceeded before weight allocation".into());
        }
        let w: Vec<f32> = (0..weights)
            .map(|j| ((j % 251) as f32 - 125.) / 257.)
            .collect();
        for tokens in [1usize, 4, 8, 16, 32] {
            let input_len = tokens.checked_mul(input).ok_or("overflow")?;
            let output_len = tokens.checked_mul(output).ok_or("overflow")?;
            let payload = weights
                .checked_add(input_len)
                .and_then(|n| output_len.checked_mul(7).and_then(|m| n.checked_add(m)))
                .and_then(|n| n.checked_mul(4))
                .ok_or("overflow")?;
            if payload > 6 * 1024 * 1024 * 1024 {
                return Err("budget exceeded".into());
            }
            let x: Vec<f32> = (0..input_len)
                .map(|i| ((i * 17 % 101) as f32 - 50.) / 53.)
                .collect();
            let mut outputs: Vec<Vec<f32>> = (0..5).map(|_| vec![0.; output_len]).collect();
            let mut times: Vec<Vec<f64>> = (0..5).map(|_| Vec::with_capacity(10)).collect();
            let calculate = |variant: usize, y: &mut [f32]| {
                if variant == 0 {
                    linear_lanes4(
                        std::hint::black_box(&x),
                        std::hint::black_box(&w),
                        y,
                        tokens,
                        input,
                        output,
                    )
                    .unwrap();
                } else {
                    linear_chunked(
                        std::hint::black_box(&x),
                        std::hint::black_box(&w),
                        y,
                        tokens,
                        input,
                        output,
                        [1, 2, 4, 8][variant - 1],
                    )
                    .unwrap();
                }
            };
            for (variant, y) in outputs.iter_mut().enumerate() {
                calculate(variant, y);
            }
            if outputs[1..].iter().any(|y| y != &outputs[0]) {
                return Err("chunk arithmetic differs from four-lane reference".into());
            }
            // FP64 numerical reference before timing, never the production kernel.
            let mut max_abs = 0.0_f64;
            for t in 0..tokens {
                for (o, row) in w.chunks_exact(input).enumerate() {
                    let expected: f64 = row
                        .iter()
                        .zip(&x[t * input..(t + 1) * input])
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let a = outputs[0][t * output + o];
                    let error = (f64::from(a) - expected).abs();
                    max_abs = max_abs.max(error);
                    if !a.is_finite() || error > 1e-5 + 1e-4 * expected.abs() {
                        return Err("frozen operator gate failed".into());
                    }
                }
            }
            for _ in 0..3 {
                for (variant, y) in outputs.iter_mut().enumerate() {
                    calculate(variant, y);
                }
            }
            for repetition in 0..10 {
                for slot in 0..5 {
                    let variant = (slot + repetition) % 5;
                    let start = Instant::now();
                    calculate(variant, &mut outputs[variant]);
                    let ns = start.elapsed().as_nanos();
                    std::hint::black_box(&outputs[variant]);
                    times[variant].push(ns as f64 / 1e6);
                    writeln!(raw,"{}",json!({"input":input,"output":output,"tokens":tokens,"variant":variant,"chunk":if variant==0{None}else{Some([1,2,4,8][variant-1])},"repetition":repetition,"elapsed_ns":ns})).map_err(|e|e.to_string())?;
                }
            }
            let mut variants = Vec::new();
            for (variant, timing) in times.iter().enumerate() {
                let chunk = if variant == 0 {
                    1
                } else {
                    [1, 2, 4, 8][variant - 1]
                };
                variants.push(json!({"variant":variant,"label":if variant==0{"independent matrix-vector rows"}else{"weight-group reuse across token chunk"},"chunk":chunk,"ms":distribution(timing)?,"logical_full_weight_passes":tokens.div_ceil(chunk)}));
            }
            summary.push(json!({"input":input,"output":output,"tokens":tokens,"payload_bytes":payload,"stack_scratch_bytes":128,"max_abs_error_fp64":max_abs,"variants":variants,"movement_scope":"pass counts are logical loop visits, not measured DRAM traffic/bandwidth"}));
        }
    }
    std::fs::write(
        path.join("summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(
        path.join("completion.json"),
        b"{\"run_status\":\"complete\"}",
    )
    .map_err(|e| e.to_string())?;
    println!("Saved fifteen projection workloads to {}", path.display());
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
