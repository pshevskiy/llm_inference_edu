// T19 public Instruments control; no engine integration or Rust boundary.
import CoreML
import Foundation
import os

let args = CommandLine.arguments
guard args.count == 3, ["tiny", "qwen"].contains(args[1]) else {
    fatalError("pass tiny|qwen COMPILED_MODEL")
}
let signposter = OSSignposter(subsystem: "org.llm-inference.t19", category: .pointsOfInterest)
print("signposterEnabled=\(signposter.isEnabled) pid=\(ProcessInfo.processInfo.processIdentifier)")
let configuration = MLModelConfiguration()
configuration.computeUnits = args[1] == "tiny" ? .cpuOnly : .cpuAndNeuralEngine
let load = signposter.beginInterval("Load")
let model = try MLModel(contentsOf: URL(fileURLWithPath: args[2]), configuration: configuration)
signposter.endInterval("Load", load)
// Qwen IDs are the three public T06 reference cases, not sampled outputs.
let cases = args[1] == "tiny" ? [[1], [1, 5], [1, 5, 7]] : [[9707, 0], [49, 590, 374], [11613, 5519, 1378]]
let qwenGreedy = [[271, 358], [16731, 374, 264], [220, 2326, 16819]]
for (caseIndex, ids) in cases.enumerated() {
    let state = args[1] == "qwen" ? model.makeState() : nil
    let steps = args[1] == "tiny" ? [ids] : ids.map { [$0] }
    for (position, tokens) in steps.enumerated() {
        let tokenArray = try MLMultiArray(shape: [NSNumber(value: tokens.count)], dataType: .int32)
        for (index, token) in tokens.enumerated() { tokenArray[index] = NSNumber(value: token) }
        var inputs: [String: Any] = ["token_ids": tokenArray]
        if state != nil {
            let positionArray = try MLMultiArray(shape: [1], dataType: .int32)
            positionArray[0] = NSNumber(value: position)
            inputs["position"] = positionArray
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let interval = signposter.beginInterval("Prediction")
        let result: any MLFeatureProvider
        if let state { result = try model.prediction(from: provider, using: state) }
        else { result = try model.prediction(from: provider) }
        signposter.endInterval("Prediction", interval)
        guard let logits = result.featureValue(for: "logits")?.multiArrayValue else {
            fatalError("missing logits")
        }
        let start = args[1] == "tiny" ? (tokens.count - 1) * 64 : 0
        let count = args[1] == "tiny" ? 64 : 151936
        var best = 0
        for index in 0..<count {
            guard logits[start + index].doubleValue.isFinite else { fatalError("nonfinite logits") }
            if logits[start + index].doubleValue > logits[start + best].doubleValue { best = index }
        }
        if args[1] == "qwen" && best != qwenGreedy[caseIndex][position] {
            fatalError("control failed frozen T06 greedy check")
        }
        print("ids=\(ids) position=\(position) greedy=\(best)")
    }
}
