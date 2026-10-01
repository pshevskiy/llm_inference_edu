use llm_inference::linear;

fn main() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = [1.0, 0.0, -1.0, 2.0, 1.0, 0.0];
    let mut y = [0.0; 4];

    linear(&x, &w, &mut y, 2, 3, 2).expect("valid example shapes");
    assert_eq!(y, [-2.0, 4.0, -2.0, 13.0]);
    println!("Y = [{:?}, {:?}]", &y[..2], &y[2..]);
}
