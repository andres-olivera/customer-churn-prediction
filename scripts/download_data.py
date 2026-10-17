"""Download the official UCI dataset and verify its basic source contract."""
from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/563/iranian%2Bchurn%2Bdataset.zip"
DEST = Path("data/Customer_Churn.csv")


def main():
    with urlopen(URL, timeout=60) as response:
        payload = response.read()
    with ZipFile(BytesIO(payload)) as archive:
        candidates = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one source CSV, got {candidates}")
        raw = archive.read(candidates[0])
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(StringIO(text))
    if "Churn" not in (reader.fieldnames or []):
        raise RuntimeError("UCI file missing Churn outcome")
    if len(list(reader)) != 3150:
        raise RuntimeError("Unexpected UCI row count; source may have changed")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(text, encoding="utf-8")
    print(f"Verified 3150 rows from UCI and wrote {DEST}")


if __name__ == "__main__":
    main()
