//! Explicit progress persistence. Validation and atomic same-directory replacement
//! happen independently of model loading, generation and volatile cache lifetime.
use super::{MAX_PROGRESS_BYTES, Progress};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};
static SAVE_ID: AtomicU64 = AtomicU64::new(0);
pub fn load(path: &Path) -> Result<Progress, String> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Progress::default()),
        Err(e) => return Err(e.to_string()),
    };
    let mut bytes = Vec::new();
    file.take((MAX_PROGRESS_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    Progress::from_json(&bytes).map_err(|e| format!("progress:{e:?}"))
}
pub fn save(path: &Path, progress: &Progress) -> Result<(), String> {
    let bytes = progress.to_json().map_err(|e| format!("progress:{e:?}"))?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path.file_name().ok_or("missing progress filename")?;
    let temp = parent.join(format!(
        ".{}-{}-{}.tmp",
        name.to_string_lossy(),
        std::process::id(),
        SAVE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temp).map_err(|e| e.to_string())?;
    let result = (|| {
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)
    })()
    .map_err(|e: std::io::Error| e.to_string());
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}
