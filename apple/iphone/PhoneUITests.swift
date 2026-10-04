import XCTest

// Run on a named iOS Simulator first; separately repeat on a signed physical
// phone. The tiny fixture's expected tokens are independent reference answers.
final class PhoneUITests: XCTestCase {
    @MainActor
    func testTokensPauseBackgroundResetAndUnload() {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launch()
        let load = app.buttons["load"]
        XCTAssertTrue(load.waitForExistence(timeout: 10))
        waitEnabled(load)
        tap(app, "load")
        waitEnabled(app.buttons["prepare"])
        tap(app, "prepare")
        waitLabel(app.staticTexts["counts"], "Evaluated: 3 · History: 3")
        tap(app, "step")
        waitLabel(app.staticTexts["tokens"], "41")
        waitLabel(app.staticTexts["counts"], "Evaluated: 3 · History: 4")

        XCUIDevice.shared.press(.home)
        app.activate()
        waitLabel(app.staticTexts["state"], "Paused")
        XCTAssertFalse(app.buttons["step"].isEnabled)
        tap(app, "resume")
        for expected in ["41, 53", "41, 53, 41", "41, 53, 41, 50"] {
            waitEnabled(app.buttons["step"])
            tap(app, "step")
            waitLabel(app.staticTexts["tokens"], expected)
        }
        tap(app, "step") // observe STOPPED; last emission remains pending
        waitLabel(app.staticTexts["state"], "Finished")
        waitLabel(app.staticTexts["counts"], "Evaluated: 6 · History: 7")
        tap(app, "pause")
        tap(app, "resume")
        waitLabel(app.staticTexts["state"], "Finished")
        XCTAssertFalse(app.buttons["step"].isEnabled)
        app.swipeUp() // session controls may be below the small phone's viewport
        tap(app, "reset")
        waitLabel(app.staticTexts["counts"], "Evaluated: 0 · History: 0")
        tap(app, "unload")
        waitLabel(app.staticTexts["state"], "Unloaded")
        XCUIDevice.shared.press(.home)
        app.activate()
        waitEnabled(load) // an empty session must re-enable loading on foreground
        tap(app, "load")
        waitEnabled(app.buttons["prepare"])
        tap(app, "unload")
        waitLabel(app.staticTexts["state"], "Unloaded")
    }

    @MainActor
    func testDebugMemoryWarningClearsAndAllowsReload() {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launch()
        waitEnabled(app.buttons["load"])
        tap(app, "load")
        waitEnabled(app.buttons["prepare"])
        tap(app, "prepare")
        waitEnabled(app.buttons["step"])
        tap(app, "step")
        waitLabel(app.staticTexts["tokens"], "41")
        waitLabel(app.staticTexts["counts"], "Evaluated: 3 · History: 4")
        tap(app, "memory-warning-demo")
        waitLabel(app.staticTexts["state"], "Unloaded")
        waitLabel(app.staticTexts["tokens"], "No emitted tokens yet")
        waitLabel(app.staticTexts["counts"], "Evaluated: 0 · History: 0")
        tap(app, "load")
        waitEnabled(app.buttons["prepare"])
        tap(app, "unload")
        waitLabel(app.staticTexts["state"], "Unloaded")
    }

    @MainActor
    private func tap(_ app: XCUIApplication, _ identifier: String) {
        let button = app.buttons[identifier]
        for _ in 0..<4 {
            if button.isHittable { break }
            if button.frame.midY < app.frame.midY { app.swipeDown() }
            else { app.swipeUp() }
        }
        XCTAssertTrue(button.isHittable, "Control is off screen: \(identifier)")
        button.tap()
    }

    @MainActor
    private func waitLabel(_ element: XCUIElement, _ expected: String) {
        let match = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label == %@", expected), object: element)
        XCTAssertEqual(XCTWaiter.wait(for: [match], timeout: 10), .completed)
    }
    @MainActor
    private func waitEnabled(_ element: XCUIElement) {
        let match = XCTNSPredicateExpectation(predicate: NSPredicate(format: "enabled == true"), object: element)
        XCTAssertEqual(XCTWaiter.wait(for: [match], timeout: 10), .completed)
    }
}
