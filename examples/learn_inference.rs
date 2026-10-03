use llm_inference::education::{EXERCISES, Progress, exercise, storage};
use std::path::Path;
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("list") if args.len() == 1 => {
            for e in EXERCISES {
                println!("{}: {}", e.id, e.question());
            }
        }
        Some("show") if args.len() == 2 => {
            let e = exercise(&args[1]).map_err(|e| format!("{e:?}"))?;
            println!("{}\n{}", e.question(), e.hint());
        }
        Some("answer") if args.len() == 3 || args.len() == 5 && args[3] == "--progress" => {
            let path = args.get(4).map(Path::new);
            let mut progress = match path {
                Some(path) => storage::load(path)?,
                None => Progress::default(),
            };
            let feedback = progress
                .answer(&args[1], &args[2])
                .map_err(|e| format!("{e:?}"))?;
            if let Some(path) = path {
                storage::save(path, &progress)?;
            }
            if feedback.correct {
                println!("Correct. {}", feedback.reviewed_explanation.unwrap_or(""));
            } else {
                println!("Try again. {}", feedback.hint);
            }
            println!(
                "Progress: {} completed exercise(s).",
                progress.entries.iter().filter(|e| e.completed).count()
            );
        }
        _ => {
            return Err(
                "usage: learn_inference list | show ID | answer ID DECIMAL [--progress PATH]"
                    .into(),
            );
        }
    }
    Ok(())
}
