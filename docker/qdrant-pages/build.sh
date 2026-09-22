#!/usr/bin/env bash
# Run on Linux / WSL. Never invoke Cargo on the Windows host.
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Run this build inside Linux/WSL, never on the Windows host' >&2; exit 1; }
[[ "$(uname -m)" == x86_64 ]] || { echo 'This demo image targets Linux x86-64' >&2; exit 1; }
root=$(cd "$(dirname "$0")/../.." && pwd)
recipe="$root/docker/qdrant-pages"
data=$(realpath -m "${KV_SEARCH_DATA:-$(dirname "$root")/kv-search-data}")
artifacts="$data/artifacts/qdrant-pages"
source_repo=$(realpath "${QDRANT_SOURCE:-$(dirname "$root")/qdrant-ood}")
[[ -f "$source_repo/lib/page-attention/src/bin/prepare.rs" ]] || {
  echo 'Set QDRANT_SOURCE to Qdrant branch page-attention-demo including the offline builder' >&2
  exit 1
}
work=$(realpath -m "${QDRANT_BUILD_ROOT:-$HOME/.cache/kv-search/qdrant-pages-build}")
tag=${1:-qdrant-pages:demo}
mkdir -p "$work"
export PATH="$HOME/.cargo/bin:$PATH"
source_dir=$(python3 "$recipe/source.py" "$source_repo" "$work")
base=$(git -C "$source_repo" rev-parse HEAD)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$work/target}
# Deliberately use the baseline x86-64 target. SIMD is selected at runtime.
export RUSTFLAGS=""
unset CARGO_ENCODED_RUSTFLAGS
export GIT_COMMIT_ID="$base"
cd "$source_dir"
cargo test --locked -p page-attention --features builder --lib -j "${BUILD_JOBS:-8}"
cargo build --locked -p page-attention --features builder --bin page-attention-prepare --profile perf -j "${BUILD_JOBS:-8}"
cargo build --locked -p qdrant --profile perf -j "${BUILD_JOBS:-8}"
context=$(mktemp -d "$work/image.XXXXXX")
cp "$CARGO_TARGET_DIR/perf/"{qdrant,page-attention-prepare} "$context/"
strip "$context/qdrant" "$context/page-attention-prepare"
cp "$recipe/"{Dockerfile,config.yaml} "$context/"
cp "$source_dir/source.json" "$context/"
cp LICENSE "$context/LICENSE"
printf '%s\n' 'Qdrant with offline page-attention preparation and serving.' \
  'Source checkout revision:' "$base" > "$context/NOTICE"
python3 - "$context" "$(rustc --version)" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
(p / 'build.json').write_text(json.dumps({
    'rustc': sys.argv[2], 'profile': 'perf', 'target_cpu': 'x86-64 (runtime SIMD dispatch)',
    'builder_sha256': hashlib.sha256((p / 'page-attention-prepare').read_bytes()).hexdigest(),
    'binary_sha256': hashlib.sha256((p / 'qdrant').read_bytes()).hexdigest(),
}, indent=2) + '\n')
PY
docker build --platform linux/amd64 --build-arg "SOURCE_REVISION=$base" \
  -t "$tag" "$context"
docker run --rm "$tag" --version
mkdir -p "$artifacts"
cp "$context/"{build,source}.json "$artifacts/"
docker image inspect "$tag" > "$artifacts/image-inspect.json"
if [[ "${EXPORT_IMAGE:-1}" == 1 ]]; then
  docker image save -o "$artifacts/qdrant-pages-demo.tar" "$tag"
fi
echo "Built $tag; provenance: /qdrant/{source,build}.json inside the image"
