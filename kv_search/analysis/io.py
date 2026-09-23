import datetime
import json
import subprocess
from pathlib import Path

ANALYSIS_DIR = Path("cache/analysis")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def envelope_path(name: str) -> Path:
    return ANALYSIS_DIR / f"{name}.json"


def read_envelope(name: str) -> dict:
    return json.loads(envelope_path(name).read_text())


def write_envelope(name: str, model: str, params: dict, sizes: dict) -> Path:
    """Standard analysis artifact: metadata + params + per-size results."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    env = {
        "analysis": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "kv_search_commit": _git_commit(),
        "params": params,
        "sizes": sizes,
    }
    path = envelope_path(name)
    path.write_text(json.dumps(env))
    return path
