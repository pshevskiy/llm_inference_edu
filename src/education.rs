//! Owner-focused row-major exercises. Curated content, deterministic grading,
//! explicit progress and untrusted model probes have separate responsibilities.
use serde::{Deserialize, Serialize};

#[path = "education_storage.rs"]
pub mod storage;
pub const MAX_PROGRESS_BYTES: usize = 4096;
pub const MAX_ATTEMPTS: u32 = 1000;
pub const PROBE_SYSTEM: &str = "Explain row-major offsets. Reply with one JSON object only. Fields: row, width, column, offset (integers), formula (exactly \"row*width+column\"), explanation (one short sentence). Do not use Markdown.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    UnknownExercise,
    ZeroDimension,
    DimensionOverflow,
    InvalidIndex,
    InvalidAnswer,
    InvalidProgress,
    AttemptLimit,
    InvalidExplanation,
}

#[derive(Debug, Clone, Copy)]
pub struct Exercise {
    pub id: &'static str,
    pub tensor: &'static str,
    pub rows: usize,
    pub width: usize,
    pub row: usize,
    pub column: usize,
    pub reviewed_explanation: &'static str,
}
pub const EXERCISES: [Exercise; 3] = [
    Exercise {
        id: "x-offset",
        tensor: "X",
        rows: 3,
        width: 4,
        row: 2,
        column: 1,
        reviewed_explanation: "Each input token has four consecutive elements. Skip two complete rows, then move one element along the third row.",
    },
    Exercise {
        id: "w-offset",
        tensor: "W",
        rows: 2,
        width: 4,
        row: 1,
        column: 2,
        reviewed_explanation: "Weights store one row per output feature. Skip one row of four weights, then move two elements within the second row.",
    },
    Exercise {
        id: "y-offset",
        tensor: "Y",
        rows: 3,
        width: 2,
        row: 1,
        column: 1,
        reviewed_explanation: "Each token has two output features. Skip the first row, then select the second feature of the next row.",
    },
];
pub fn exercise(id: &str) -> Result<&'static Exercise, Error> {
    EXERCISES
        .iter()
        .find(|e| e.id == id)
        .ok_or(Error::UnknownExercise)
}
pub fn row_major_offset(
    rows: usize,
    width: usize,
    row: usize,
    column: usize,
) -> Result<usize, Error> {
    if rows == 0 || width == 0 {
        return Err(Error::ZeroDimension);
    }
    rows.checked_mul(width).ok_or(Error::DimensionOverflow)?;
    if row >= rows || column >= width {
        return Err(Error::InvalidIndex);
    }
    row.checked_mul(width)
        .and_then(|base| base.checked_add(column))
        .ok_or(Error::DimensionOverflow)
}
impl Exercise {
    pub fn question(&self) -> String {
        format!(
            "{} is row-major [{},{}]. Explain the element offset of {}[{},{}].",
            self.tensor, self.rows, self.width, self.tensor, self.row, self.column
        )
    }
    pub fn hint(&self) -> String {
        format!(
            "Each row contains {} elements. Count whole rows first, then columns. Use zero-based indices; answer in elements, not bytes.",
            self.width
        )
    }
    fn answer(&self) -> Result<usize, Error> {
        row_major_offset(self.rows, self.width, self.row, self.column)
    }
}
fn decimal(text: &str) -> Result<usize, Error> {
    if text.is_empty() || text.len() > 20 || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Error::InvalidAnswer);
    }
    text.parse().map_err(|_| Error::InvalidAnswer)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub id: String,
    pub attempts: u32,
    pub correct: u32,
    pub completed: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progress {
    pub version: u32,
    pub entries: Vec<Entry>,
}
impl Default for Progress {
    fn default() -> Self {
        Self {
            version: 1,
            entries: Vec::new(),
        }
    }
}
pub struct Feedback {
    pub correct: bool,
    pub hint: String,
    pub reviewed_explanation: Option<&'static str>,
}
impl Progress {
    pub fn validate(&self) -> Result<(), Error> {
        if self.version != 1 || self.entries.len() > EXERCISES.len() {
            return Err(Error::InvalidProgress);
        }
        let mut total = 0_u32;
        for (i, e) in self.entries.iter().enumerate() {
            if exercise(&e.id).is_err()
                || self.entries[..i].iter().any(|prior| prior.id == e.id)
                || e.attempts == 0
                || e.correct > e.attempts
                || e.completed != (e.correct > 0)
            {
                return Err(Error::InvalidProgress);
            }
            total = total
                .checked_add(e.attempts)
                .ok_or(Error::InvalidProgress)?;
        }
        if total > MAX_ATTEMPTS {
            return Err(Error::InvalidProgress);
        }
        Ok(())
    }
    pub fn from_json(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > MAX_PROGRESS_BYTES {
            return Err(Error::InvalidProgress);
        }
        let p: Self = serde_json::from_slice(bytes).map_err(|_| Error::InvalidProgress)?;
        p.validate()?;
        Ok(p)
    }
    pub fn to_json(&self) -> Result<Vec<u8>, Error> {
        self.validate()?;
        let bytes = serde_json::to_vec_pretty(self).map_err(|_| Error::InvalidProgress)?;
        if bytes.len() > MAX_PROGRESS_BYTES {
            return Err(Error::InvalidProgress);
        }
        Ok(bytes)
    }
    /// Validate loaded state and the entire answer before changing any counter.
    /// Never stores answer text, prompts, explanations, model identity or KV state.
    pub fn answer(&mut self, id: &str, text: &str) -> Result<Feedback, Error> {
        self.validate()?;
        let problem = exercise(id)?;
        let answer = decimal(text)?;
        let correct = answer == problem.answer()?;
        if self.entries.iter().map(|e| e.attempts).sum::<u32>() >= MAX_ATTEMPTS {
            return Err(Error::AttemptLimit);
        }
        let index = if let Some(i) = self.entries.iter().position(|e| e.id == id) {
            i
        } else {
            self.entries.push(Entry {
                id: id.into(),
                attempts: 0,
                correct: 0,
                completed: false,
            });
            self.entries.len() - 1
        };
        let e = &mut self.entries[index];
        e.attempts += 1;
        if correct {
            e.correct += 1;
            e.completed = true;
        }
        Ok(Feedback {
            correct,
            hint: problem.hint(),
            reviewed_explanation: correct.then_some(problem.reviewed_explanation),
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedFacts {
    pub row: usize,
    pub width: usize,
    pub column: usize,
    pub offset: usize,
    pub formula: String,
    pub explanation: String,
}
/// Experimental factual gate only: matching fields does not validate prose or
/// authorize model wording as a reviewed lesson. Normal learning uses curation.
pub fn check_generated_facts(id: &str, text: &str) -> Result<GeneratedFacts, Error> {
    if text.len() > 4096 {
        return Err(Error::InvalidExplanation);
    }
    let e = exercise(id)?;
    let facts: GeneratedFacts =
        serde_json::from_str(text).map_err(|_| Error::InvalidExplanation)?;
    if facts.row != e.row
        || facts.width != e.width
        || facts.column != e.column
        || facts.offset != e.answer()?
        || facts.formula != "row*width+column"
        || facts.explanation.trim().is_empty()
        || facts.explanation.len() > 512
    {
        return Err(Error::InvalidExplanation);
    }
    Ok(facts)
}
