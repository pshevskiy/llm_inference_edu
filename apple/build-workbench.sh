#!/bin/sh
# Optional first argument is an existing pinned Qwen directory. No downloads.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_root"
app_dir="$task_root/target/native/InferenceWorkbench.app"
resources="$app_dir/Contents/Resources"
executable="$app_dir/Contents/MacOS/InferenceWorkbench"
mkdir -p "$resources" "$app_dir/Contents/MacOS"
cargo build --offline --release -p llm_cpu_bridge
xcrun swiftc -target arm64-apple-macosx26.5 -swift-version 6 -warnings-as-errors -O -g \
  -D WORKBENCH -parse-as-library -I apple/cpu_bridge/include \
  apple/CPUClient.swift apple/BoundarySmoke.swift apple/HubSearch.swift \
  apple/ModelInstallation.swift apple/WorkbenchStateSmoke.swift apple/WorkbenchApp.swift \
  target/release/libllm_cpu_bridge.a -o "$executable"
xcrun swiftc -target arm64-apple-macosx26.5 -swift-version 6 -warnings-as-errors -O \
  -parse-as-library apple/BuildMetadata.swift -o "$task_root/target/native/build-metadata"
cp fixtures/tiny-model.json fixtures/qwen-inventory.json "$resources/"
cp target/release/libllm_cpu_bridge.a "$resources/"
printf '%s\n' "$task_root" > "$resources/workspace.txt"
cat > "$app_dir/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>dev.local.llm-inference-workbench</string>
<key>CFBundleName</key><string>Inference Workbench</string>
<key>CFBundleExecutable</key><string>InferenceWorkbench</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>LSMinimumSystemVersion</key><string>26.5</string>
<key>NSPrincipalClass</key><string>NSApplication</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
# Failures leave a provisional unqualified identity, never a stale passed one.
"$task_root/target/native/build-metadata" "$resources" "$resources/libllm_cpu_bridge.a" "$executable" 0
"$executable" --smoke tiny "$resources/tiny-model.json"
"$executable" --state-smoke
if [ "$#" -gt 0 ]; then
  "$executable" --smoke qwen "$1"
  "$task_root/target/native/build-metadata" "$resources" "$resources/libllm_cpu_bridge.a" "$executable" 1
fi
printf 'Built %s\n' "$app_dir"
