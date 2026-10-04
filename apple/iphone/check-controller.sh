#!/bin/sh
# Mac-only integration check; never reported as iOS execution.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$task_root"
mkdir -p target/native
cargo build --offline --locked --release -p llm_cpu_bridge
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swiftc \
  -target arm64-apple-macosx26.5 -swift-version 6 -warnings-as-errors \
  -D PHONE_SESSION_SMOKE -parse-as-library -O -g -module-cache-path "$task_root/target/phone-host-modules" \
  -I apple/cpu_bridge/include apple/CPUClient.swift apple/iphone/PhoneSession.swift \
  apple/iphone/PhoneSessionSmoke.swift target/release/libllm_cpu_bridge.a \
  -o target/native/phone-session-smoke
target/native/phone-session-smoke fixtures/tiny-model.json
