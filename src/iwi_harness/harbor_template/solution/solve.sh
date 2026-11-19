#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$here"
export PATH="$here/bin:$PATH"
artifacts=${IWI_ARTIFACT_DIR:-/logs/artifacts}
demo_dir=${IWI_DEMO_DIR:-/tmp/iwi-demo}
iwi-harness demo --out "$demo_dir"
mkdir -p "$artifacts"
cp "$demo_dir/patch.safetensors" "$artifacts/patch.safetensors"
