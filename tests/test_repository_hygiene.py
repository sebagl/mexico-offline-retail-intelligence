"""Repository-level guarantees: approved hosts only, no contact data, attribution present."""

import json
import re
from pathlib import Path

import pytest

from app.schemas import TRANSFORMATION_NOTICE
from app.services.dataset import ESTABLISHMENTS_FILE, FORBIDDEN_FIELDS, MANIFEST_FILE

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", ".pytest_cache", ".ruff_cache", ".cache", ".claude"}
SOURCE_SUFFIXES = {
    ".py",
    ".md",
    ".html",
    ".js",
    ".css",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
    ".example",
    "",
}
# Every hostname the repository may reference. Anything else (a scraped site, a
# leftover URL from another project, a tracking domain) fails this test.
ALLOWED_HOSTS = {
    "www.inegi.org.mx",
    "inegi.org.mx",
    "github.com",
    "render.com",
    "dashboard.render.com",
    "mexico-offline-retail-intelligence.onrender.com",
    "docs.docker.com",
    "errors.pydantic.dev",
    "claude.com",
    "127.0.0.1",
    "localhost",
    "testserver",
    "evil.example",  # used by tests as a deliberately unsafe host
    "www.example.invalid",
    "example.invalid",
}
HOST = re.compile(r"https?://([A-Za-z0-9.-]+)|\bwww\.[A-Za-z0-9.-]+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)\d{10}(?!\d)")


def _repo_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in SKIP_DIRS for part in path.parts)
        and path.suffix in SOURCE_SUFFIXES
        and not path.name.startswith(".env")
        and path.parent != ROOT / "data"
    ]


def test_only_approved_hosts_are_referenced() -> None:
    offenders = []
    for path in _repo_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in HOST.finditer(text):
            host = (match.group(1) or match.group(0)).lower().rstrip(".")
            if host not in ALLOWED_HOSTS:
                offenders.append(f"{path.relative_to(ROOT)}: {host}")
    assert offenders == []


def test_frontend_has_no_unsafe_dom_sinks() -> None:
    script = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in script
    assert "document.write" not in script
    assert "eval(" not in script


def test_attribution_and_notice_are_in_the_page() -> None:
    page = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "Fuente: INEGI, Directorio Estadístico Nacional de Unidades Económicas (DENUE)" in page
    assert TRANSFORMATION_NOTICE in page
    assert "logo" not in page.lower()  # no INEGI (or any other) logo is embedded


@pytest.mark.skipif(not (ROOT / "data" / MANIFEST_FILE).is_file(), reason="production dataset not generated")
def test_production_dataset_is_minimized_and_attributed() -> None:
    manifest = json.loads((ROOT / "data" / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["source_url"] == "https://www.inegi.org.mx/servicios/api_denue.html"
    assert manifest["attribution"].startswith(
        "Fuente: INEGI, Directorio Estadístico Nacional de Unidades Económicas (DENUE)"
    )
    assert manifest["transformation_notice"] == TRANSFORMATION_NOTICE
    payload = json.loads((ROOT / "data" / ESTABLISHMENTS_FILE).read_text(encoding="utf-8"))
    records = payload["establishments"]
    assert records
    for record in records[:2000]:
        assert not (FORBIDDEN_FIELDS & {k.lower() for k in record})
        assert "latitude" not in record and "longitude" not in record
        blob = " ".join(str(v) for v in record.values())
        assert not EMAIL.search(blob), record["id"]
        assert not PHONE.search(blob), record["id"]
        assert "TEST ESTABLISHMENT" not in record["name"]
