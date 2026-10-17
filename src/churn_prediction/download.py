"""Obtain the UCI Iranian Churn CSV from its canonical ZIP archive.

No model, GUI or CLI depends on the repository's non-packaged scripts folder.
"""
from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

UCI_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/563/iranian%2Bchurn%2Bdataset.zip"
)
DEFAULT_CSV = Path("data/Customer_Churn.csv")


def extract_source_csv(archive_bytes: bytes) -> str:
    """Decode and sanity-check a single CSV embedded in the UCI ZIP."""
    with ZipFile(BytesIO(archive_bytes)) as archive:
        csv_names = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
        ]
        if len(csv_names) != 1:
            raise ValueError(f"Expected exactly one CSV in UCI archive, got {csv_names}")
        original = archive.read(csv_names[0]).decode("utf-8-sig")
    reader = csv.DictReader(StringIO(original))
    columns = reader.fieldnames or []
    if "Churn" not in columns or "Call  Failure" not in columns:
        raise ValueError(f"Unexpected source columns: {columns}")
    count = sum(1 for _ in reader)
    if count != 3150:
        raise ValueError(f"Expected 3150 UCI rows; found {count}")
    return original


def download_data(destination: str | Path = DEFAULT_CSV) -> Path:
    """Fetch the public dataset, validate it, then write a local CSV."""
    with urlopen(UCI_ARCHIVE_URL, timeout=60) as response:
        archive_bytes = response.read()
    csv_text = extract_source_csv(archive_bytes)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(csv_text, encoding="utf-8")
    return destination
