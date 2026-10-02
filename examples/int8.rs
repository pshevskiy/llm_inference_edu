use llm_inference::linear;
use llm_inference::quantization::{GroupwiseInt8, linear_int8};
fn main() {
    let weights = [127., 0.5, -0.5, 0., 254., 0., 0., 0., 0., 0.];
    let q = GroupwiseInt8::quantize(&weights, 2, 5, 4).unwrap();
    let x = [0.5, 2., -1., 4., 0.25];
    let mut reconstructed = [0.; 10];
    q.dequantize(&mut reconstructed).unwrap();
    let mut reference = [0.; 2];
    linear(&x, &reconstructed, &mut reference, 1, 5, 2).unwrap();
    let mut fused = [0.; 2];
    linear_int8(&x, &q, &mut fused, 1).unwrap();
    assert_eq!(fused, [130., 0.]);
    assert_eq!(reference, fused);
    println!(
        "INT8 weights={:?}, row-group scales={:?}",
        q.values(),
        q.scales()
    );
    println!("fused={fused:?}, reconstructed reference={reference:?}");
    println!(
        "logical encoded payload={} bytes vs FP32={} bytes (not RSS or file size)",
        q.logical_bytes(),
        weights.len() * 4
    );
}
