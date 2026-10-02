use llm_inference::kv_cache::CachedSession;
use llm_inference::model::Decoder;
use std::time::Instant;
fn main() {
    let model = Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap();
    let ids = [1, 5, 7, 41, 53, 41, 50, 2];
    let start = Instant::now();
    let expected = model.forward(&ids, false).unwrap().logits;
    println!("tiny uncached full-prefix: {:?}", start.elapsed());
    let mut session = CachedSession::new(&model, 8).unwrap();
    let start = Instant::now();
    assert_eq!(session.prefill(&ids).unwrap(), expected);
    println!("tiny serial prefill single call: {:?}", start.elapsed());
    session.reset();
    let start = Instant::now();
    session.prefill(&ids[..3]).unwrap();
    session.append(&ids[3..5]).unwrap();
    assert_eq!(session.append(&ids[5..]).unwrap(), expected);
    println!("tiny serial chunk calls [3,2,3]: {:?}", start.elapsed());
    session.reset();
    let start = Instant::now();
    let mut last = Vec::new();
    for id in ids {
        last = session.step(id).unwrap();
    }
    assert_eq!(last, expected);
    println!("tiny explicit token steps: {:?}", start.elapsed());
    println!("All final logits exactly equal; single demonstration timings, not optimized prefill");
}
