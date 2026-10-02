use llm_inference::sampling::{Sampler, Sampling, SamplingError};
fn random() -> Sampler {
    Sampler::new(Sampling::Temperature {
        temperature: 1.,
        top_k: 2,
        seed: 42,
    })
    .unwrap()
}
#[test]
fn greedy_and_top_k_have_explicit_tie_and_membership_rules() {
    let mut greedy = Sampler::new(Sampling::Greedy).unwrap();
    assert_eq!(greedy.sample(&[2., 2., 1.]).unwrap(), 0);
    let mut top = Sampler::new(Sampling::Temperature {
        temperature: 1.,
        top_k: 1,
        seed: 42,
    })
    .unwrap();
    assert_eq!(top.sample(&[-0., 0.]).unwrap(), 0);
    for _ in 0..100 {
        assert!(random().sample(&[-10., 0., 3., 4.]).unwrap() >= 2);
    }
}
#[test]
fn seeded_sampling_is_reproducible_and_invalid_values_do_not_advance_rng() {
    let mut a = random();
    let mut b = random();
    assert_eq!(a.sample(&[f32::NAN, 0.]), Err(SamplingError::NonFinite));
    let sequence: Vec<_> = (0..64).map(|_| a.sample(&[0., 1.]).unwrap()).collect();
    assert_eq!(
        sequence,
        (0..64)
            .map(|_| b.sample(&[0., 1.]).unwrap())
            .collect::<Vec<_>>()
    );
    a.reset();
    assert_eq!(
        sequence,
        (0..64)
            .map(|_| a.sample(&[0., 1.]).unwrap())
            .collect::<Vec<_>>()
    );
    assert!(sequence.contains(&0) && sequence.contains(&1));
}
#[test]
fn sampling_mass_matches_the_independent_two_class_equation() {
    let mut sampler = random();
    let mut count = 0;
    // exp(ln(3))/(exp(0)+exp(ln(3))) = 3/4.
    for _ in 0..10000 {
        count += usize::from(sampler.sample(&[0., 3.0_f32.ln()]).unwrap() == 1);
    }
    assert!((7350..=7650).contains(&count), "count {count}");
}
#[test]
fn bad_parameters_empty_and_nonfinite_logits_are_rejected() {
    for temperature in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(matches!(
            Sampler::new(Sampling::Temperature {
                temperature,
                top_k: 1,
                seed: 0
            }),
            Err(SamplingError::InvalidParameters)
        ));
    }
    assert!(matches!(
        Sampler::new(Sampling::Temperature {
            temperature: 1.,
            top_k: 0,
            seed: 0
        }),
        Err(SamplingError::InvalidParameters)
    ));
    assert_eq!(random().sample(&[]), Err(SamplingError::EmptyLogits));
    assert_eq!(
        random().sample(&[0.]),
        Err(SamplingError::InvalidParameters)
    );
    for bad in [f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(random().sample(&[bad, 0.]), Err(SamplingError::NonFinite));
    }
    let mut sampler = Sampler::new(Sampling::Temperature {
        temperature: f32::from_bits(1),
        top_k: 2,
        seed: 42,
    })
    .unwrap();
    assert_eq!(sampler.sample(&[-f32::MAX, f32::MAX]).unwrap(), 1);
}
