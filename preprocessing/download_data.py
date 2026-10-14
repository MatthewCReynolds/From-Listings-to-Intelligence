"""Download and verify the realistic-v2 dataset into data/."""

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

URL = "https://accolade-columbia-capstone.t3.tigrisfiles.io/realistic-v2.zip"
SHA256 = "2324542388fbc6e68a915703a457158e91a81fd2abe0dc32fa3f6698c14e5894"

DATA = Path(__file__).resolve().parent.parent / "data"
ZIP = DATA / "realistic-v2.zip"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if (DATA / "realistic-v2").is_dir():
        print("dataset already present; nothing to do")
        return 0

    if not ZIP.exists():
        print(f"downloading {URL} (869 MB)")
        subprocess.run(["curl", "-L", "--fail", "-o", str(ZIP), URL], check=True)

    print("verifying checksum")
    actual = digest(ZIP)
    if actual != SHA256:
        sys.exit(f"checksum mismatch\n  expected {SHA256}\n  actual   {actual}")

    print("unpacking")
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(DATA)

    print(f"done -> {DATA / 'realistic-v2'}")
    print("next: python preprocessing/build_duckdb.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
