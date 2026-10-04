import SwiftUI
import UIKit

@main
struct InferencePhoneApp: App {
    @StateObject private var session = PhoneSession(fixture: Bundle.main.url(forResource: "tiny-model", withExtension: "json")!)
    var body: some Scene {
        WindowGroup { PhoneView(session: session) }
    }
}

private struct PhoneView: View {
    @ObservedObject var session: PhoneSession
    @Environment(\.scenePhase) private var phase
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    Text("Learn the engine, one token at a time.").font(.title2.bold())
                    Text("This small bundled model demonstrates our Rust inference engine. It is a numerical teaching fixture, not a conversational assistant.")
                        .foregroundStyle(.secondary)
                    GroupBox("1. Load and prepare") {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Prompt token IDs: 1, 5, 7").font(.system(.body, design: .monospaced))
                            Button("Load demo", action: session.load).disabled(!session.canLoad).accessibilityIdentifier("load")
                            Button("Prepare prompt", action: session.prepare).disabled(!session.canPrepare).accessibilityIdentifier("prepare")
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                    GroupBox("2. Observe generation") {
                        VStack(alignment: .leading, spacing: 12) {
                            Text(session.tokens.isEmpty ? "No emitted tokens yet" : session.tokens.map(String.init).joined(separator: ", "))
                                .font(.system(.title3, design: .monospaced)).accessibilityIdentifier("tokens")
                            Text(verbatim: "Evaluated: \(session.processed) · History: \(session.history)").accessibilityIdentifier("counts")
                            Button("Next token", action: session.step).disabled(!session.canStep).accessibilityIdentifier("step")
                            HStack {
                                Button("Pause", action: session.pause).disabled(!session.canPause).accessibilityIdentifier("pause")
                                Button("Resume", action: session.resume).disabled(!session.paused || session.busy || !session.ready).accessibilityIdentifier("resume")
                            }
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                    GroupBox("Session") {
                        VStack(alignment: .leading, spacing: 12) {
                            HStack { Text(session.state).bold().accessibilityIdentifier("state"); if session.busy { ProgressView() } }
                            Text(session.message).font(.callout).foregroundStyle(.secondary)
                            HStack {
                                Button("Reset", action: session.reset).disabled(!session.canReset).accessibilityIdentifier("reset")
                                Button("Unload") { session.unload() }.disabled(!session.hasSession).accessibilityIdentifier("unload")
                            }
                            #if DEBUG
                            Button("Memory-warning demo") {
                                // Exercise the public notification listener, not OS pressure.
                                NotificationCenter.default.post(name: UIApplication.didReceiveMemoryWarningNotification, object: nil)
                            }.disabled(!session.hasSession).accessibilityIdentifier("memory-warning-demo")
                            #endif
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                    Text("Why can history be one token longer? Emission records a token before the next forward pass evaluates it. Backgrounding pauses work; returning does not resume automatically.")
                        .font(.callout).foregroundStyle(.secondary)
                    Text("Real-model iPhone support awaits separate memory and numerical qualification. Simulator results do not certify a physical phone.")
                        .font(.caption).foregroundStyle(.secondary)
                }.padding()
            }.navigationTitle("Inference Lab")
        }
        .onChange(of: phase, initial: true) { _, value in session.sceneActive(value == .active) }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in session.memoryWarning() }
    }
}
