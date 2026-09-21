#!/usr/bin/env bash
# Run on Linux / WSL. Never invoke Cargo on the Windows host.
set -euo pipefail
[[ "$(uname -m)" == x86_64 ]] || { echo 'This demo image targets Linux x86-64' >&2; exit 1; }
root=$(cd "$(dirname "$0")/../.." && pwd)
recipe="$root/docker/qdrant-pages"
data=$(realpath -m "${KV_SEARCH_DATA:-$(dirname "$root")/kv-search-data}")
port="$data/docker/qdrant-pages"
artifacts="$data/artifacts/qdrant-pages"
[[ -f "$port/source.json" && -f "$port/page-attention.patch" ]] || {
  echo "Missing Qdrant source bundle in $port; set KV_SEARCH_DATA to the external data directory" >&2
  exit 1
}
work=${QDRANT_BUILD_ROOT:-$HOME/.cache/kv-search/qdrant-pages-build}
tag=${1:-qdrant-pages:demo}
mkdir -p "$work"
export PATH="$HOME/.cargo/bin:$PATH"
base=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_commit"])' "$port/source.json")
patch_hash=$(sha256sum "$port/page-attention.patch" | cut -d' ' -f1)
expected_hash=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["patch_sha256"])' "$port/source.json")
[[ "$patch_hash" == "$expected_hash" ]] || { echo 'Source patch checksum mismatch' >&2; exit 1; }
source_dir="$work/source-$patch_hash"
if [[ ! -f "$source_dir/.prepared" ]]; then
  # A new directory per patch; never overwrite an existing source checkout.
  staging=$(mktemp -d "$work/source.XXXXXX")
  if [[ -n "${QDRANT_BASE_REPO:-}" ]]; then
    git -C "$QDRANT_BASE_REPO" archive "$base" | tar -x -C "$staging"
  else
    curl --fail --location --retry 3 "https://codeload.github.com/qdrant/qdrant/tar.gz/$base" \
      | tar -xz --strip-components=1 -C "$staging"
  fi
  git -C "$staging" apply --check "$port/page-attention.patch"
  git -C "$staging" apply "$port/page-attention.patch"
  touch "$staging/.prepared"
  [[ ! -e "$source_dir" ]] || { echo "Incomplete source already exists: $source_dir" >&2; exit 1; }
  mv "$staging" "$source_dir"
fi
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$work/target}
# Deliberately use the baseline x86-64 target. SIMD is selected at runtime.
export RUSTFLAGS=""
unset CARGO_ENCODED_RUSTFLAGS
export GIT_COMMIT_ID="$base"
cd "$source_dir"
cargo test --locked -p page-attention --lib -j "${BUILD_JOBS:-8}"
cargo build --locked -p qdrant --profile perf -j "${BUILD_JOBS:-8}"
context=$(mktemp -d "$work/image.XXXXXX")
cp "$CARGO_TARGET_DIR/perf/qdrant" "$context/qdrant"
strip "$context/qdrant"
cp "$recipe/"{Dockerfile,config.yaml} "$context/"
cp "$port/source.json" "$context/"
cp LICENSE "$context/LICENSE"
printf '%s\n' 'Page-attention source patch is distributed in the kv-search-data bundle.' \
  'Based on Qdrant OOD commit:' "$base" > "$context/NOTICE"
python3 - "$context" "$(rustc --version)" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
(p / 'build.json').write_text(json.dumps({
    'rustc': sys.argv[2], 'profile': 'perf', 'target_cpu': 'x86-64 (runtime SIMD dispatch)',
    'binary_sha256': hashlib.sha256((p / 'qdrant').read_bytes()).hexdigest(),
}, indent=2) + '\n')
PY
docker build --platform linux/amd64 --build-arg "SOURCE_REVISION=$base" \
  --build-arg "PATCH_SHA256=$patch_hash" -t "$tag" "$context"
docker run --rm "$tag" --version
mkdir -p "$artifacts"
cp "$context/build.json" "$artifacts/build.json"
docker image inspect "$tag" > "$artifacts/image-inspect.json"
if [[ "${EXPORT_IMAGE:-1}" == 1 ]]; then
  docker image save -o "$artifacts/qdrant-pages-demo.tar" "$tag"
fi
echo "Built $tag; provenance: /qdrant/{source,build}.json inside the image"
