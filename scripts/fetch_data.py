#!/usr/bin/env python
"""Download the released answers (a Hugging Face dataset, configs/release.json) into the data folder and check every
file against configs/data_manifest.json (rows are not re-counted; sha256 is). Files already present with the right
sha256 are skipped, so the script also verifies a folder you copied by hand.

    python scripts/fetch_data.py               # into $QR_DATA, else local.json {"data": ...}, else ../questioning-retrieval-data
    python scripts/fetch_data.py --verify      # check only, download nothing
    python scripts/fetch_data.py --only docci_s1 winoground

Plain HTTPS from the dataset's public URL; no Hugging Face account or library is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qr.data import data_root   # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="check the local files only")
    ap.add_argument("--only", nargs="*", help="sources (folders) to fetch; default all")
    a = ap.parse_args()
    rel = json.loads((ROOT / "configs/release.json").read_text())
    man = json.loads((ROOT / "configs/data_manifest.json").read_text())
    out = data_root()
    files = [x for k, v in man.items() if not a.only or k in a.only for x in v]
    bad = fetched = 0
    for x in files:
        f = out / x["file"]
        if f.exists() and sha256(f) == x["sha256"]:
            continue
        if a.verify:
            print(f"{'wrong sha256' if f.exists() else 'missing'}: {x['file']}")
            bad += 1
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        url = f"{rel['base_url']}/{x['file']}"
        print(f"downloading {x['file']} ({x['bytes'] / 1e6:.1f} MB)", flush=True)
        tmp = f.with_suffix(f.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        if sha256(tmp) != x["sha256"]:
            print(f"   sha256 mismatch for {x['file']}; kept as {tmp.name}")
            bad += 1
            continue
        tmp.replace(f)
        fetched += 1
    n = len(files)
    print(f"{out}: {n - bad} of {n} files verified" + (f" ({fetched} downloaded)" if fetched else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
