#!/bin/sh
# Used by Xcode. No target/toolchain/component installation or signing changes.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
case "${PLATFORM_NAME:-}" in
  iphonesimulator) task_target=aarch64-apple-ios-sim ;;
  iphoneos) task_target=aarch64-apple-ios ;;
  *) printf 'Unsupported platform: %s\n' "${PLATFORM_NAME:-missing}" >&2; exit 1 ;;
esac
case "${ARCHS:-arm64}" in
  arm64) ;;
  *) printf 'Unsupported architectures: %s. This slice targets arm64 only.\n' "$ARCHS" >&2; exit 1 ;;
esac
task_rustc=$(rustup which --toolchain 1.90 rustc)
task_cargo=$(rustup which --toolchain 1.90 cargo)
task_sysroot=$("$task_rustc" --print sysroot)
if [ ! -d "$task_sysroot/lib/rustlib/$task_target/lib" ]; then
  printf 'Not built: Rust target missing. Explicit setup required: rustup target add --toolchain 1.90 %s\n' "$task_target" >&2
  exit 1
fi
# Simulator and physical-device libraries must never share an architecture-only
# path; they both use arm64 but have distinct Mach-O platform identities.
cd "$task_root"
# Cargo also compiles build scripts/procedural macros for the Mac host. Xcode's
# exported iOS SDKROOT makes their native linker look for the wrong libSystem.
# Use the Mac SDK as the environment default; rustc and cc select the explicit
# iOS target SDK independently (and ignore a SDKROOT from the wrong platform).
task_host_sdk=$(xcrun --sdk macosx --show-sdk-path)
SDKROOT="$task_host_sdk" RUSTC="$task_rustc" CARGO_TARGET_DIR="$task_root/target/iphone-rust" \
  IPHONEOS_DEPLOYMENT_TARGET=17.0 "$task_cargo" build --offline --locked \
  --release -p llm_cpu_bridge --target "$task_target"
mkdir -p "$task_root/target/iphone/$PLATFORM_NAME"
cp "$task_root/target/iphone-rust/$task_target/release/libllm_cpu_bridge.a" \
  "$task_root/target/iphone/$PLATFORM_NAME/libllm_cpu_bridge.a"
