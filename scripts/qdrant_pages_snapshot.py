"""Export or restore a Qdrant collection snapshot without overwriting a collection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import requests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["export", "restore"])
    ap.add_argument("file", type=Path)
    ap.add_argument("--url", default="http://localhost:6433")
    ap.add_argument("--collection", default="pages_100k")
    ap.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY"))
    args = ap.parse_args()
    session = requests.Session()
    if args.api_key:
        session.headers["api-key"] = args.api_key
    endpoint = f"{args.url.rstrip('/')}/collections/{args.collection}"
    if args.action == "export":
        if args.file.exists():
            raise SystemExit(f"Refusing to overwrite {args.file}")
        response = session.post(endpoint + "/snapshots", timeout=1800)
        response.raise_for_status()
        name = response.json()["result"]["name"]
        args.file.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with session.get(endpoint + f"/snapshots/{name}", stream=True, timeout=1800) as response:
            response.raise_for_status()
            with args.file.open("xb") as output:
                for block in response.iter_content(8 * 1024 * 1024):
                    output.write(block)
                    digest.update(block)
        sha = digest.hexdigest()
        args.file.with_suffix(args.file.suffix + ".sha256").write_text(f"{sha}  {args.file.name}\n")
        print(json.dumps({"file": str(args.file), "bytes": args.file.stat().st_size, "sha256": sha}))
    else:
        check = session.get(endpoint, timeout=30)
        if check.status_code != 404:
            check.raise_for_status()
            raise SystemExit(f"Collection {args.collection!r} exists; choose a fresh name")
        checksum = args.file.with_suffix(args.file.suffix + ".sha256")
        if checksum.exists():
            with args.file.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != checksum.read_text().split()[0]:
                raise SystemExit("Snapshot checksum mismatch")
        # curl streams multipart uploads; requests' files= buffers multi-GB files.
        # Keep credentials out of the subprocess command line.
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "curl.conf"
            config.write_text("" if not args.api_key else
                              "header = " + json.dumps("api-key: " + args.api_key) + "\n")
            config.chmod(0o600)
            subprocess.run(["curl", "--fail-with-body", "--silent", "--show-error",
                            "--config", str(config), "--max-time", "1800",
                            "-X", "POST", "-F", f"snapshot=@{args.file.resolve()}",
                            endpoint + "/snapshots/upload?priority=snapshot"], check=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            response = session.get(endpoint, timeout=30)
            response.raise_for_status()
            if response.json()["result"]["status"] == "green":
                print(f"\nRestored {args.collection}")
                return
            time.sleep(1)
        raise SystemExit("Snapshot uploaded, but collection is not green yet")


if __name__ == "__main__":
    main()
