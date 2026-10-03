use llm_inference::education::{self, Entry, Error, Progress, storage};
#[test]
fn offsets_and_checked_indices() {
    assert_eq!(education::row_major_offset(3, 4, 2, 1), Ok(9));
    assert_eq!(education::row_major_offset(2, 4, 1, 2), Ok(6));
    assert_eq!(education::row_major_offset(3, 2, 2, 1), Ok(5));
    assert_eq!(
        education::row_major_offset(3, 4, 3, 0),
        Err(Error::InvalidIndex)
    );
    assert_eq!(
        education::row_major_offset(3, 4, 0, 4),
        Err(Error::InvalidIndex)
    );
    assert_eq!(
        education::row_major_offset(0, 4, 0, 0),
        Err(Error::ZeroDimension)
    );
    assert_eq!(
        education::row_major_offset(usize::MAX, 2, 0, 0),
        Err(Error::DimensionOverflow)
    );
}
#[test]
fn grading_retries_and_invalid_preservation() {
    let mut p = Progress::default();
    let wrong = p.answer("x-offset", "8").unwrap();
    assert!(!wrong.correct);
    assert!(wrong.reviewed_explanation.is_none());
    for answer in ["", "-1", " 9", "+9", "9.0", "99999999999999999999"] {
        let prior = p.clone();
        assert!(p.answer("x-offset", answer).is_err());
        assert_eq!(p, prior);
    }
    assert!(p.answer("x-offset", "9").unwrap().correct);
    assert!(p.answer("w-offset", "6").unwrap().correct);
    assert!(p.answer("y-offset", "3").unwrap().correct);
    assert_eq!(p.entries[0].attempts, 2);
    assert_eq!(p.entries[0].correct, 1);
    assert_eq!(Progress::from_json(&p.to_json().unwrap()).unwrap(), p);
    let prior = p.clone();
    assert!(p.answer("unknown", "0").is_err());
    assert_eq!(p, prior);
}
#[test]
fn loaded_state_and_budget_are_checked_before_mutation() {
    let base = Entry {
        id: "x-offset".into(),
        attempts: 1,
        correct: 0,
        completed: false,
    };
    let invalid = [
        Progress {
            version: 2,
            entries: vec![],
        },
        Progress {
            version: 1,
            entries: vec![base.clone(), base.clone()],
        },
        Progress {
            version: 1,
            entries: vec![Entry {
                completed: true,
                ..base.clone()
            }],
        },
        Progress {
            version: 1,
            entries: vec![Entry {
                correct: 2,
                ..base.clone()
            }],
        },
        Progress {
            version: 1,
            entries: vec![Entry {
                id: "unknown".into(),
                ..base.clone()
            }],
        },
    ];
    for mut p in invalid {
        let prior = p.clone();
        assert_eq!(
            p.answer("x-offset", "9").err(),
            Some(Error::InvalidProgress)
        );
        assert_eq!(p, prior);
    }
    let mut p = Progress {
        version: 1,
        entries: vec![Entry {
            attempts: 1000,
            ..base
        }],
    };
    let prior = p.clone();
    assert_eq!(p.answer("x-offset", "9").err(), Some(Error::AttemptLimit));
    assert_eq!(p, prior);
    assert!(Progress::from_json(br#"{"version":1,"entries":[],"unknown":0}"#).is_err());
    assert!(Progress::from_json(&vec![b' '; 4097]).is_err());
}
#[test]
fn model_fields_are_untrusted_and_not_grading_authority() {
    let valid = r#"{"row":2,"width":4,"column":1,"offset":9,"formula":"row*width+column","explanation":"Two whole rows, then one column."}"#;
    assert!(education::check_generated_facts("x-offset", valid).is_ok());
    for invalid in [
        valid.replace("\"offset\":9", "\"offset\":8"),
        valid.replace("row*width+column", "row+column"),
        valid.replace("Two whole rows, then one column.", ""),
        valid.replace("\"row\":2", "\"row\":3"),
        valid.replace("\"row\":2", "\"unknown\":2,\"row\":2"),
        valid.replace("\"row\":2", "\"row\":2,\"row\":2"),
        format!("{valid} trailing"),
        "x".repeat(4097),
    ] {
        assert!(education::check_generated_facts("x-offset", &invalid).is_err());
    }
    // Plausible fields cannot establish prose truth: normal app uses curation.
    assert!(
        education::check_generated_facts(
            "x-offset",
            &valid.replace("Two whole rows, then one column.", "The Moon is cheese.")
        )
        .is_ok()
    );
}
#[test]
fn explicit_atomic_progress_save_preserves_prior_on_rejected_state() {
    let dir = std::env::temp_dir().join(format!("llm-education-test-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("progress.json");
    let mut p = Progress::default();
    assert_eq!(storage::load(&path).unwrap(), p);
    p.answer("x-offset", "8").unwrap();
    storage::save(&path, &p).unwrap();
    let old = std::fs::read(&path).unwrap();
    let invalid = Progress {
        version: 2,
        entries: vec![],
    };
    assert!(storage::save(&path, &invalid).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), old);
    p.answer("x-offset", "9").unwrap();
    storage::save(&path, &p).unwrap();
    assert_eq!(storage::load(&path).unwrap(), p);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    std::fs::remove_dir_all(dir).unwrap();
}
