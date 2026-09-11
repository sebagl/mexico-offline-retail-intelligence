"""Artifact loading: manifest validation, checksums, forbidden fields, model match."""

import json
from pathlib import Path

import pytest

from app.exceptions import DatasetError
from app.services.dataset import (
    AGGREGATES_FILE,
    ESTABLISHMENTS_FILE,
    KNOWLEDGE_FILE,
    MANIFEST_FILE,
    load_dataset,
    write_bytes_atomic,
)
from tests.conftest import FAKE_MODEL_NAME


def _rewrite(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_valid_dataset_loads(data_dir: Path) -> None:
    dataset = load_dataset(data_dir, FAKE_MODEL_NAME)
    assert dataset.record_count == 41
    assert dataset.knowledge.matrix.shape[0] == dataset.knowledge.document_count
    assert dataset.manifest.source_url.startswith("https://www.inegi.org.mx/")
    assert not hasattr(dataset.knowledge.documents[0], "embedding")  # vectors live only in the matrix


def test_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="missing data artifact"):
        load_dataset(tmp_path, FAKE_MODEL_NAME)


def test_model_mismatch(data_dir: Path) -> None:
    with pytest.raises(DatasetError, match="EMBEDDING_MODEL"):
        load_dataset(data_dir, "another-model")


def test_checksum_mismatch_detected(data_dir: Path) -> None:
    _rewrite(data_dir / ESTABLISHMENTS_FILE, lambda p: p["establishments"].pop())
    with pytest.raises(DatasetError, match="checksum"):
        load_dataset(data_dir, FAKE_MODEL_NAME)


def test_aggregate_tampering_detected(data_dir: Path) -> None:
    _rewrite(data_dir / AGGREGATES_FILE, lambda p: p.__setitem__("total", 999))
    with pytest.raises(DatasetError, match="aggregates do not match"):
        load_dataset(data_dir, FAKE_MODEL_NAME)


def test_forbidden_contact_field_rejected(data_dir: Path) -> None:
    path = data_dir / ESTABLISHMENTS_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["establishments"][0]["Telefono"] = "5555555555"
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    write_bytes_atomic(path, raw.encode("utf-8"))
    from app.services.dataset import checksum_bytes

    _rewrite(
        data_dir / MANIFEST_FILE,
        lambda p: p.__setitem__("data_checksum", checksum_bytes(raw.encode("utf-8"))),
    )
    with pytest.raises(DatasetError, match=r"schema error|disallowed"):
        load_dataset(data_dir, FAKE_MODEL_NAME)


def test_manifest_schema_error(data_dir: Path) -> None:
    _rewrite(data_dir / MANIFEST_FILE, lambda p: p.pop("attribution"))
    with pytest.raises(DatasetError, match=r"manifest\.json schema error"):
        load_dataset(data_dir, FAKE_MODEL_NAME)


def test_knowledge_dimension_mismatch(data_dir: Path) -> None:
    _rewrite(data_dir / KNOWLEDGE_FILE, lambda p: p["documents"][0]["embedding"].append(0.1))
    with pytest.raises(DatasetError, match="dimension"):
        load_dataset(data_dir, FAKE_MODEL_NAME)


def test_knowledge_nan_rejected(data_dir: Path) -> None:
    def mutate(p: dict) -> None:
        p["documents"][0]["embedding"][0] = float("nan")

    _rewrite(data_dir / KNOWLEDGE_FILE, mutate)
    with pytest.raises(DatasetError):
        load_dataset(data_dir, FAKE_MODEL_NAME)


def test_malformed_json_is_controlled(data_dir: Path) -> None:
    (data_dir / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="not readable JSON"):
        load_dataset(data_dir, FAKE_MODEL_NAME)


def test_atomic_write_replaces_file(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    write_bytes_atomic(target, b"first")
    write_bytes_atomic(target, b"second")
    assert target.read_bytes() == b"second"
    assert list(tmp_path.iterdir()) == [target]
