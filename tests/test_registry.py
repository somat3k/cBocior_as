"""
tests/test_registry.py — Unit tests for the ModelRegistry (K3+K4+K8).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.registry import (
    ModelRegistry,
    compute_file_hash,
    verify_file_hash,
)

# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------

class TestComputeFileHash:
    def test_returns_hex_string(self, tmp_path: Path) -> None:
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h = compute_file_hash(f)
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"identical")
        f2.write_bytes(b"identical")
        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_missing_file_returns_empty_string(self, tmp_path: Path) -> None:
        h = compute_file_hash(tmp_path / "nonexistent.bin")
        assert h == ""


class TestVerifyFileHash:
    def test_correct_hash_returns_true(self, tmp_path: Path) -> None:
        f = tmp_path / "file.bin"
        f.write_bytes(b"test content")
        h = compute_file_hash(f)
        assert verify_file_hash(f, h) is True

    def test_wrong_hash_returns_false(self, tmp_path: Path) -> None:
        f = tmp_path / "file.bin"
        f.write_bytes(b"test content")
        assert verify_file_hash(f, "a" * 64) is False

    def test_empty_expected_returns_true(self, tmp_path: Path) -> None:
        f = tmp_path / "file.bin"
        f.write_bytes(b"any")
        # Empty expected → skip check, return True
        assert verify_file_hash(f, "") is True


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

@pytest.fixture()
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(export_dir=tmp_path)


def _make_files(tmp_path: Path, symbol: str, tf: str) -> dict[str, Path]:
    """Create dummy model files for testing."""
    sym_dir = tmp_path / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, suffix in [("nn", ".npz"), ("gbm", ".joblib"),
                         ("scaler", ".joblib"), ("features", ".joblib")]:
        p = sym_dir / f"{symbol}_{tf}_{key}{suffix}"
        p.write_bytes(b"dummy content " + key.encode())
        files[key] = p
    return files


class TestModelRegistryRecord:
    def test_creates_registry_json(self, registry: ModelRegistry, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "EURUSD", "M1")
        registry.record("EURUSD", "M1", files)
        assert (tmp_path / "registry.json").exists()

    def test_creates_metadata_json(self, registry: ModelRegistry, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "EURUSD", "M1")
        registry.record("EURUSD", "M1", files)
        meta_path = tmp_path / "EURUSD" / "EURUSD_M1_meta.json"
        assert meta_path.exists()

    def test_metadata_contains_hashes(self, registry: ModelRegistry, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "EURUSD", "M1")
        meta = registry.record("EURUSD", "M1", files)
        for key in ("nn", "gbm", "scaler", "features"):
            assert meta["files"][key]["sha256"] != ""

    def test_version_increments_on_repeat_record(
        self, registry: ModelRegistry, tmp_path: Path
    ) -> None:
        files = _make_files(tmp_path, "EURUSD", "M1")
        meta1 = registry.record("EURUSD", "M1", files)
        meta2 = registry.record("EURUSD", "M1", files)
        assert meta2["version"] == meta1["version"] + 1

    def test_metrics_stored_in_registry(self, registry: ModelRegistry, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "BTCUSD", "H1")
        metrics = {"nn_val_acc": 0.62, "bt_win_rate": 55.5, "bt_max_drawdown_pct": 3.1}
        meta = registry.record("BTCUSD", "H1", files, metrics=metrics)
        assert meta["nn_val_acc"] == pytest.approx(0.62)
        assert meta["bt_win_rate"] == pytest.approx(55.5)

    def test_multiple_symbols_stored_independently(
        self, registry: ModelRegistry, tmp_path: Path
    ) -> None:
        files_eu = _make_files(tmp_path, "EURUSD", "M1")
        files_btc = _make_files(tmp_path, "BTCUSD", "H1")
        registry.record("EURUSD", "M1", files_eu)
        registry.record("BTCUSD", "H1", files_btc)

        all_entries = registry.all_entries()
        assert "EURUSD" in all_entries
        assert "BTCUSD" in all_entries

    def test_registry_json_is_valid(self, registry: ModelRegistry, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "EURUSD", "M1")
        registry.record("EURUSD", "M1", files)
        reg_path = tmp_path / "registry.json"
        loaded = json.loads(reg_path.read_text())
        assert "EURUSD" in loaded
        assert "M1" in loaded["EURUSD"]


class TestModelRegistryVerify:
    def test_verify_passes_for_fresh_export(
        self, registry: ModelRegistry, tmp_path: Path
    ) -> None:
        files = _make_files(tmp_path, "EURUSD", "M1")
        registry.record("EURUSD", "M1", files)
        assert registry.verify("EURUSD", "M1") is True

    def test_verify_fails_after_file_tampered(
        self, registry: ModelRegistry, tmp_path: Path
    ) -> None:
        files = _make_files(tmp_path, "EURUSD", "M1")
        registry.record("EURUSD", "M1", files)
        # Tamper with one file
        files["gbm"].write_bytes(b"tampered!")
        assert registry.verify("EURUSD", "M1") is False

    def test_verify_returns_false_for_unknown_symbol(
        self, registry: ModelRegistry
    ) -> None:
        assert registry.verify("UNKNOWN", "M1") is False

    def test_verify_fails_when_file_deleted(
        self, registry: ModelRegistry, tmp_path: Path
    ) -> None:
        files = _make_files(tmp_path, "EURUSD", "M1")
        registry.record("EURUSD", "M1", files)
        files["nn"].unlink()
        assert registry.verify("EURUSD", "M1") is False


class TestModelRegistryHelpers:
    def test_get_entry_returns_none_for_missing(self, registry: ModelRegistry) -> None:
        assert registry.get_entry("MISSING", "M1") is None

    def test_get_entry_returns_meta(self, registry: ModelRegistry, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "EURUSD", "M5")
        registry.record("EURUSD", "M5", files)
        entry = registry.get_entry("EURUSD", "M5")
        assert entry is not None
        assert entry["symbol"] == "EURUSD"

    def test_list_symbols_empty_initially(self, registry: ModelRegistry) -> None:
        assert registry.list_symbols() == []

    def test_list_symbols_after_record(self, registry: ModelRegistry, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "EURUSD", "M1")
        registry.record("EURUSD", "M1", files)
        assert "EURUSD" in registry.list_symbols()

    def test_corrupt_registry_starts_fresh(self, registry: ModelRegistry, tmp_path: Path) -> None:
        reg_path = tmp_path / "registry.json"
        reg_path.write_text("{invalid json", encoding="utf-8")
        # Should not raise; returns fresh empty dict
        assert registry.all_entries() == {}
