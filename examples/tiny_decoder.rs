use llm_inference::model::Decoder;

fn main() {
    let decoder = Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap();
    let mut ids = vec![1, 5, 7];
    for _ in 0..4 {
        let output = decoder.forward(&ids, false).unwrap();
        // Strict greater-than preserves the lowest token ID on a tie.
        let mut next = 0;
        for i in 1..output.logits.len() {
            if output.logits[i] > output.logits[next] {
                next = i;
            }
        }
        println!("prefix {ids:?} -> token {next}");
        ids.push(next);
    }
    println!("Synthetic tokens only; these weights do not model a language.");
}
