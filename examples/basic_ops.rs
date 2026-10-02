use llm_inference::ops::{embedding, residual_add, rmsnorm, silu, softmax};

fn main() {
    let mut embedded = [0.0; 2];
    embedding(&[1], &[1.0, 0.0, 3.0, 4.0], &mut embedded, 2, 2).unwrap();
    println!("embedding: {embedded:?}");

    let mut added = [0.0; 2];
    residual_add(&embedded, &[1.0, -2.0], &mut added).unwrap();
    println!("residual:  {added:?}");

    let mut normalized = [0.0; 2];
    rmsnorm(&added, &[1.0; 2], &mut normalized, 1, 2, 1e-5).unwrap();
    println!("RMSNorm:   {normalized:?}");

    let mut activated = [0.0; 2];
    silu(&normalized, &mut activated).unwrap();
    println!("SiLU:      {activated:?}");

    let mut probabilities = [0.0; 2];
    softmax(&activated, &mut probabilities, 1, 2).unwrap();
    println!("softmax:   {probabilities:?}");
}
