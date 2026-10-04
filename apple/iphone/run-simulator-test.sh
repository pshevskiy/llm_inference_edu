#!/bin/sh
# Cargo runner for existing numerical tests. Explicit device, no auto boot/setup.
set -eu
: "${INFERENCE_SIMULATOR_UUID:?Set an explicit booted Simulator UUID}"
DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}" \
  exec xcrun simctl spawn "$INFERENCE_SIMULATOR_UUID" "$@"
