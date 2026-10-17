"""Offline tests for the hosted-app UCI download path."""
from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from churn_prediction import download


def source_zip(*, rows: int = 3150, include_target: bool = True) -> bytes:
    out = StringIO()
    out.write("Call  Failure,Churn\n" if include_target else "Call  Failure,Wrong\n")
    out.writelines("8,0\n" for _ in range(rows))
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        archive.writestr("Customer Churn.csv", out.getvalue())
    return buffer.getvalue()


def test_extract_official_shape():
    content = download.extract_source_csv(source_zip())
    assert content.startswith("Call  Failure,Churn")
    assert len(content.strip().splitlines()) == 3151


def test_reject_missing_churn():
    with pytest.raises(ValueError, match="Unexpected source"):
        download.extract_source_csv(source_zip(include_target=False))


def test_reject_changed_row_count():
    with pytest.raises(ValueError, match="Expected 3150"):
        download.extract_source_csv(source_zip(rows=2))


def test_download_uses_importable_package(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        download, "urlopen", lambda *args, **kwargs: BytesIO(source_zip())
    )
    destination = tmp_path / "source" / "customers.csv"
    result = download.download_data(destination)
    assert result == destination
    assert result.is_file()
    assert len(result.read_text(encoding="utf-8").strip().splitlines()) == 3151


def test_streamlit_app_does_not_import_nonpackage_scripts():
    source = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text(
        encoding="utf-8"
    )
    assert "from scripts." not in source
    assert "from churn_prediction.download import download_data" in source
