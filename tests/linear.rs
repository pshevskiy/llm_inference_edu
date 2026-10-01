use llm_inference::{LinearError, linear};

#[test]
fn required_example() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = [1.0, 0.0, -1.0, 2.0, 1.0, 0.0];
    let mut y = [99.0; 4];
    linear(&x, &w, &mut y, 2, 3, 2).unwrap();
    assert_eq!(y, [-2.0, 4.0, -2.0, 13.0]);
}

#[test]
fn single_token() {
    let mut y = [99.0; 2];
    linear(&[2.0, -1.0], &[3.0, 4.0, -2.0, 5.0], &mut y, 1, 2, 2).unwrap();
    // 2*3 + (-1)*4 = 2; 2*(-2) + (-1)*5 = -9.
    assert_eq!(y, [2.0, -9.0]);
}

#[test]
fn nonsquare_dimensions() {
    let x = [1.0, 2.0, 3.0, 4.0, -1.0, 5.0];
    let w = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
    let mut y = [99.0; 12];
    linear(&x, &w, &mut y, 3, 2, 4).unwrap();
    // Each row becomes [a, b, a+b, 2*a-b], calculated by hand.
    assert_eq!(
        y,
        [1.0, 2.0, 3.0, 0.0, 3.0, 4.0, 7.0, 2.0, -1.0, 5.0, 4.0, -7.0]
    );
}

#[test]
fn negative_inputs_and_weights() {
    let mut y = [99.0; 4];
    linear(
        &[-2.0, 3.0, 1.0, -4.0],
        &[-1.0, 2.0, 3.0, -2.0],
        &mut y,
        2,
        2,
        2,
    )
    .unwrap();
    assert_eq!(y, [8.0, -12.0, -9.0, 11.0]);
}

#[test]
fn zero_weights_overwrite_output() {
    let mut y = [99.0; 4];
    linear(
        &[1.0, -2.0, 3.0, 4.0, 5.0, -6.0],
        &[0.0; 6],
        &mut y,
        2,
        3,
        2,
    )
    .unwrap();
    assert_eq!(y, [0.0; 4]);
}

#[test]
fn short_and_long_buffers_leave_output_unchanged() {
    // Check each buffer independently, both below and above its exact length.
    for (buffer, x_len, w_len, y_len) in [
        ("x", 5, 6, 4),
        ("x", 7, 6, 4),
        ("w", 6, 5, 4),
        ("w", 6, 7, 4),
        ("y", 6, 6, 3),
        ("y", 6, 6, 5),
    ] {
        let x = vec![1.0; x_len];
        let w = vec![1.0; w_len];
        let mut y = vec![37.0; y_len];
        let before = y.clone();
        let (expected, actual) = match buffer {
            "x" => (6, x_len),
            "w" => (6, w_len),
            "y" => (4, y_len),
            _ => unreachable!(),
        };
        assert_eq!(
            linear(&x, &w, &mut y, 2, 3, 2),
            Err(LinearError::InvalidLength {
                buffer,
                expected,
                actual
            }),
        );
        assert_eq!(y, before, "invalid {buffer} length modified output");
    }
}

#[test]
fn zero_dimensions_leave_output_unchanged() {
    for (tokens, input_dim, output_dim) in [(0, 3, 2), (2, 0, 2), (2, 3, 0), (0, 0, 0)] {
        let mut y = [37.0; 4];
        assert_eq!(
            linear(&[], &[], &mut y, tokens, input_dim, output_dim),
            Err(LinearError::ZeroDimension),
        );
        assert_eq!(y, [37.0; 4]);
    }
}

#[test]
fn each_shape_product_overflow_leaves_output_unchanged() {
    // Overflow X's length, W's length, and Y's length, respectively.
    for (tokens, input_dim, output_dim) in
        [(usize::MAX, 2, 1), (1, usize::MAX, 2), (usize::MAX, 1, 2)]
    {
        let mut y = [37.0; 4];
        assert_eq!(
            linear(&[], &[], &mut y, tokens, input_dim, output_dim),
            Err(LinearError::DimensionOverflow),
        );
        assert_eq!(y, [37.0; 4]);
    }
}
