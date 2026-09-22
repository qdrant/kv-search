"""Copy a content-addressed source snapshot to Linux build storage (no external patch)."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

if sys.platform != "linux":
    raise SystemExit("Run source preparation inside Linux/WSL")
repo, work = (Path(arg).resolve() for arg in sys.argv[1:])
files = sorted(set(subprocess.check_output(
    ["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
).decode().split("\0")) - {""})
staging = Path(tempfile.mkdtemp(prefix="source.", dir=work))
digest = hashlib.sha256()
for name in files:
    source = repo / name
    if not source.exists():
        continue
    target = staging / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    digest.update(name.encode() + b"\0")
    with target.open("rb") as f:
        digest.update(hashlib.file_digest(f, "sha256").digest())
revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
provenance = {"revision": revision, "source_tree_sha256": digest.hexdigest()}
(staging / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
destination = work / f"source-{revision[:12]}-{digest.hexdigest()}"
if destination.exists():
    shutil.rmtree(staging)  # our own mkdtemp directory under the explicit build root
else:
    staging.rename(destination)
print(destination)
