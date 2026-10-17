"""Acquire the verified ISOT mirror into ignored local storage.

The exact archive hash prevents an unnoticed dataset change. No credentials or
paid service are required for the anonymous endpoint verified on 17 Oct 2026.
"""
import argparse
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

URL = "https://www.kaggle.com/api/v1/datasets/download/clmentbisaillon/fake-and-real-news-dataset"
EXPECTED_SHA256 = "38bf7e046c3a59b2c746c38f2fb74f4603493c6ed9c997f2f2ddb3969705fb42"
FILES = {
    "True.csv": "ba0844414a65dc6ae7402b8eee5306da24b6b56488d6767135af466c7dcb2775",
    "Fake.csv": "bebf8bcfe95678bf2c732bf413a2ce5f621af0102c82bf08083b2e5d3c693d0c",
}


def main():
    """Reuse matching cached files, otherwise fetch and verify before writing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    destination = parser.parse_args().data_dir
    destination.mkdir(parents=True, exist_ok=True)
    if all((destination / name).is_file() and
           hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
           for name, digest in FILES.items()):
        print("Verified local CSV cache; no network request needed.")
        return
    print("Downloading public ISOT mirror (about 43 MB)...", flush=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "headline-analyzer-academic-project/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Archive differs from verified version. Keep the cached dataset; review the source before replacing it.")
    with zipfile.ZipFile(io.BytesIO(archive)) as dataset:
        if dataset.testzip() is not None:
            raise ValueError("Archive integrity check failed")
        verified = {name: dataset.read(name) for name in FILES}
    for name, contents in verified.items():
        if hashlib.sha256(contents).hexdigest() != FILES[name]:
            raise ValueError(f"Unexpected file checksum: {name}")
    for name, contents in verified.items():
        (destination / name).write_bytes(contents)
    print("Verified True.csv and Fake.csv saved to data/raw/ (excluded from Git).")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Dataset acquisition failed: {error}") from error
