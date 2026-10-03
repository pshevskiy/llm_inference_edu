//! Opt-in, bounded operation qualification; no model load or decoder integration.
#[cfg(target_os = "macos")]
mod macos {
    use llm_metal_slice::{Error, MAX_ELEMENTS, ResidualAdd};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{path::Path, time::Instant};

    fn digest(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
    fn command(program: &str, args: &[&str]) -> Option<String> {
        let result = std::process::Command::new(program)
            .args(args)
            .output()
            .ok()?;
        result
            .status
            .success()
            .then(|| String::from_utf8_lossy(&result.stdout).trim().to_string())
    }
    fn save(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
        Ok(())
    }
    fn gate(actual: &[f32], expected: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
        let mut budget = 0_f32;
        for (&a, &b) in actual.iter().zip(expected) {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite result".into());
            }
            budget = budget.max((a - b).abs() / (1e-6 + 1e-6 * b.abs()));
        }
        if budget > 1.0 {
            return Err(format!("frozen residual gate failed: {budget}").into());
        }
        Ok(budget)
    }
    fn require(condition: bool, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        if condition {
            Ok(())
        } else {
            Err(message.into())
        }
    }
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        if cfg!(debug_assertions) {
            return Err("release build required".into());
        }
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.len() != 2 || args[0] != "--run" {
            return Err("usage: residual --run FRESH_EVIDENCE_DIRECTORY".into());
        }
        let output = Path::new(&args[1]);
        if output.exists() {
            return Err("use a fresh evidence directory".into());
        }
        std::fs::create_dir_all(output)?;
        let binary = std::fs::read(std::env::current_exe()?)?;
        let mut source = Sha256::new();
        for file in [
            "Cargo.lock",
            "apple/metal_slice/Cargo.toml",
            "apple/metal_slice/src/lib.rs",
            "apple/metal_slice/src/macos.rs",
            "apple/metal_slice/src/residual.metal",
            "apple/metal_slice/examples/residual.rs",
            "src/ops.rs",
        ] {
            let bytes = std::fs::read(file)?;
            source.update((file.len() as u64).to_le_bytes());
            source.update(file.as_bytes());
            source.update((bytes.len() as u64).to_le_bytes());
            source.update(bytes);
        }
        save(
            &output.join("identity.json"),
            &json!({"status":"started","source_sha256":format!("{:x}",source.finalize()),"executable_sha256":digest(&binary),
            "host":{"arch":std::env::consts::ARCH,"os":std::env::consts::OS,"version":command("sw_vers",&["-productVersion"]),"build":command("sw_vers",&["-buildVersion"]),"chip":command("sysctl",&["-n","machdep.cpu.brand_string"]),"memory_bytes":command("sysctl",&["-n","hw.memsize"]),"rustc":command("rustc",&["-vV"]),"developer_directory":command("xcode-select",&["-p"])},
            "operation":"finite FP32 residual_add","metal_binding":"0.33.0","fast_math":false,"atol":1e-6,"rtol":1e-6,"capacity":MAX_ELEMENTS,"logical_shared_buffers_bytes":12*MAX_ELEMENTS,
            "workloads":[1,31,32,33,255,256,257,4097,262144,MAX_ELEMENTS],"warmups":3,"samples":10,"subnormal_policy":"absolute gate applies; IEEE subnormal preservation not promised"}),
        )?;
        let result = qualify(output);
        match &result {
            Ok(()) => save(&output.join("completion.json"), &json!({"status":"passed"}))?,
            Err(error) => save(
                &output.join("completion.json"),
                &json!({"status":"failed","error":error.to_string()}),
            )?,
        }
        result
    }
    fn qualify(output: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();
        let mut gpu = ResidualAdd::new(MAX_ELEMENTS)?;
        let setup_ns = start.elapsed().as_nanos();
        let mut cases = Vec::new();
        // Hand-derived answers do not call production scalar code.
        let mut y = [91.0; 4];
        gpu.run(&[1.0, -2.0, 0.0, 4.0], &[2.0, 3.0, -5.0, -4.0], &mut y)?;
        cases.push(json!({"case":"hand-derived","result":y}));
        save(&output.join("progress.json"), &json!({"cases":cases}))?;
        require(y == [3.0, 1.0, -5.0, 0.0], "hand-derived output mismatch")?;
        let x = [1e20, -0.0, 0.0, f32::from_bits(1), f32::MIN_POSITIVE];
        let residual = [
            -1e20,
            -0.0,
            -0.0,
            f32::from_bits(1),
            -f32::MIN_POSITIVE / 2.0,
        ];
        let mut actual = [91.0; 5];
        let mut expected = [0.0; 5];
        llm_inference::ops::residual_add(&x, &residual, &mut expected)
            .map_err(|e| format!("scalar {e:?}"))?;
        gpu.run(&x, &residual, &mut actual)?;
        cases.push(json!({"case":"cancellation-signed-zero-subnormal","expected_bits":expected.map(f32::to_bits),"actual_bits":actual.map(f32::to_bits)}));
        save(&output.join("progress.json"), &json!({"cases":cases}))?;
        gate(&actual, &expected)?;
        require(
            (0..3).all(|i| actual[i].to_bits() == expected[i].to_bits()),
            "cancellation/signed-zero mismatch",
        )?;
        // Every invalid call preserves caller-owned output, then executor reuse
        // must work; numerical overflow is detected after GPU completion.
        let mut sentinel = [91.0; 2];
        let invalid = [
            (&[][..], &[][..]),
            (&[1.0, 2.0][..], &[3.0][..]),
            (&[1.0][..], &[2.0][..]),
            (&[f32::NAN, 1.0][..], &[2.0, 3.0][..]),
            (&[1.0, 2.0][..], &[f32::INFINITY, 3.0][..]),
            (&[f32::MAX, 1.0][..], &[f32::MAX, 1.0][..]),
        ];
        for (index, (x, residual)) in invalid.iter().enumerate() {
            sentinel = [91.0; 2];
            let result = gpu.run(x, residual, &mut sentinel);
            cases.push(json!({"case":"invalid","index":index,"result":format!("{result:?}"),"output":sentinel}));
            save(&output.join("progress.json"), &json!({"cases":cases}))?;
            require(
                result.is_err() && sentinel == [91.0; 2],
                "invalid input accepted or output modified",
            )?;
            if index == 5 {
                require(
                    matches!(
                        result,
                        Err(Error::NonFinite {
                            buffer: "GPU output",
                            ..
                        })
                    ),
                    "expected finite-input arithmetic overflow error",
                )?;
            }
        }
        gpu.run(&[1.0, 2.0], &[3.0, 4.0], &mut sentinel)?;
        require(sentinel == [4.0, 6.0], "reuse after invalid call failed")?;
        let mut small = ResidualAdd::new(2)?;
        let mut three = [91.0; 3];
        let result = small.run(&[1.0; 3], &[2.0; 3], &mut three);
        require(
            matches!(result, Err(Error::TooLarge)) && three == [91.0; 3],
            "capacity rejection/output preservation failed",
        )?;
        drop(small);
        cases.push(json!({"case":"reuse-and-capacity","result":"passed"}));
        save(&output.join("progress.json"), &json!({"cases":cases}))?;
        let mut rows = Vec::new();
        for n in [1, 31, 32, 33, 255, 256, 257, 4097, 262144, MAX_ELEMENTS] {
            let x: Vec<f32> = (0..n).map(|i| (i % 511) as f32 * 0.125 - 31.875).collect();
            let residual: Vec<f32> = (0..n).map(|i| (i % 503) as f32 * 0.25 - 62.75).collect();
            let mut expected = vec![0.0; n];
            let mut actual = vec![91.0; n];
            llm_inference::ops::residual_add(&x, &residual, &mut expected)
                .map_err(|e| format!("scalar {e:?}"))?;
            let mut workload = Vec::with_capacity(n * 8);
            for value in x.iter().chain(&residual) {
                workload.extend_from_slice(&value.to_le_bytes());
            }
            let workload_hash = digest(&workload);
            drop(workload);
            for _ in 0..3 {
                gpu.run(&x, &residual, &mut actual)?;
            }
            let mut samples = Vec::new();
            let mut max_budget = 0_f32;
            for _ in 0..10 {
                let start = Instant::now();
                let timing = gpu.run(&x, &residual, &mut actual)?;
                let total_ns = start.elapsed().as_nanos();
                max_budget = max_budget.max(gate(&actual, &expected)?);
                let start = Instant::now();
                llm_inference::ops::residual_add(
                    std::hint::black_box(&x),
                    std::hint::black_box(&residual),
                    std::hint::black_box(&mut expected),
                )
                .map_err(|e| format!("scalar {e:?}"))?;
                let scalar_ns = start.elapsed().as_nanos();
                std::hint::black_box(&expected);
                samples.push(json!({"public_call_ns":total_ns,"submission_wait_ns":timing.submission_wait_ns,"gpu_seconds":timing.seconds,"scalar_ns":scalar_ns}));
            }
            println!("passed residual n={n}, budget={max_budget}");
            rows.push(json!({"elements":n,"workload_sha256":workload_hash,"max_budget":max_budget,"samples":samples}));
            save(
                &output.join("progress.json"),
                &json!({"cases":cases,"workloads":rows}),
            )?;
        }
        save(
            &output.join("result.json"),
            &json!({"status":"passed","device":gpu.device_name(),"unified_memory":gpu.has_unified_memory(),"group_width":gpu.group_width(),"capacity":gpu.capacity(),"setup_ns":setup_ns,"cases":cases,"workloads":rows,
            "timing_scope":"one bounded process; 3 warmups/10 samples per length; public call includes validation and CPU copies; submission/wait and GPU timestamps reported separately; scalar call includes its validation; not full model performance",
            "physical_memory":"see permitted /usr/bin/time -l process log; logical buffers are not process footprint","energy":"unavailable","decoder_integration":"not implemented; CPU engine remains production path"}),
        )?;
        Ok(())
    }
}
#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("not run: this opt-in operation requires macOS and a Metal device");
    std::process::exit(1);
}
