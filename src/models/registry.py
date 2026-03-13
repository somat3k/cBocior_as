"""
src/models/registry.py — Model versioning, metadata, and integrity checks.

Implements:
  K2 — Keep last 3 model versions; auto-archive older ones.
  K3 — Metadata JSON alongside each exported model artefact.
  K4 — Central ``exports/registry.json`` tracking all model versions.
  K8 — SHA-256 file-hash computation and verification for integrity checks.

Registry schema (``exports/registry.json``)::

    {
      "EURUSD": {
        "M1": {
          "version": 3,
          ...
        }
      }
    }

Archive layout (``exports/archive/<SYMBOL>/``)::

    EURUSD_M1_v1_nn.npz
    EURUSD_M1_v1_model.joblib
    ...
    EURUSD_M1_v1_meta.json
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from constants import MODEL_EXPORT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REGISTRY_FILENAME = "registry.json"
_KEEP_VERSIONS = 3          # K2: keep this many recent versions per symbol/TF
_ARCHIVE_DIR = "archive"    # subdirectory under export_dir for old versions


# ---------------------------------------------------------------------------
# Hash utilities  (K8)
# ---------------------------------------------------------------------------

def compute_file_hash(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65_536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        logger.warning("Could not hash file", path=str(path), error=str(exc))
        return ""


def verify_file_hash(path: Path, expected: str) -> bool:
    """
    Return *True* when the file's SHA-256 matches *expected*.

    An empty *expected* string disables the check (returns True) so that
    registries created before hash recording was introduced still load.
    """
    if not expected:
        return True
    actual = compute_file_hash(path)
    if actual != expected:
        logger.error(
            "Model integrity check FAILED",
            path=str(path),
            expected=expected[:12] + "…",
            actual=actual[:12] + "…",
        )
        return False
    return True


# ---------------------------------------------------------------------------
# ModelRegistry  (K3 + K4)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Manages model metadata files and the central registry.

    Parameters
    ----------
    export_dir : Path
        Root export directory (default: ``MODEL_EXPORT_DIR``).
    """

    def __init__(self, export_dir: Path = MODEL_EXPORT_DIR) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.export_dir / _REGISTRY_FILENAME
        self._archive_dir = self.export_dir / _ARCHIVE_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        symbol: str,
        timeframe: str,
        files: dict[str, Path],
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Record a newly exported model artefact set.

        Parameters
        ----------
        symbol : str
        timeframe : str
        files : dict[str, Path]
            Keys: ``"nn"``, ``"gbm"``, ``"scaler"``, ``"features"``.
            Values: absolute paths to the exported artefacts.
        metrics : dict, optional
            Training + backtest metrics (``nn_val_acc``, ``gbm_val_acc``,
            ``bt_win_rate``, etc.).

        Returns
        -------
        dict  — the metadata dict written to the JSON file.
        """
        registry = self._load_registry()

        # Determine version: one higher than the previous entry for this key
        existing = registry.get(symbol, {}).get(timeframe, {})
        version = int(existing.get("version", 0)) + 1

        # K2: archive the previous version before overwriting
        if existing:
            self._archive_version(symbol, timeframe, existing)

        # Compute hashes and build file index
        file_index: dict[str, dict[str, str]] = {}
        for key, path in files.items():
            if path and path.exists():
                sha256 = compute_file_hash(path)
                rel_path = path.relative_to(self.export_dir).as_posix()
                file_index[key] = {"path": rel_path, "sha256": sha256}

        meta: dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "version": version,
            "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            **(metrics or {}),
            "files": file_index,
        }

        # Write per-model metadata JSON  (K3)
        symbol_dir = self.export_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        meta_path = symbol_dir / f"{symbol}_{timeframe}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Update central registry  (K4)
        registry.setdefault(symbol, {})[timeframe] = meta
        self._save_registry(registry)

        logger.info(
            "Model recorded in registry",
            symbol=symbol,
            timeframe=timeframe,
            version=version,
        )
        return meta

    def verify(
        self,
        symbol: str,
        timeframe: str,
    ) -> bool:
        """
        Verify the integrity of all model files for *symbol* / *timeframe*.  (K8)

        Returns *True* only when every registered file exists and its
        SHA-256 matches the stored hash.
        """
        registry = self._load_registry()
        entry = registry.get(symbol, {}).get(timeframe)
        if not entry:
            logger.warning(
                "No registry entry found for symbol/timeframe",
                symbol=symbol,
                timeframe=timeframe,
            )
            return False

        all_ok = True
        for key, info in entry.get("files", {}).items():
            path = self.export_dir / info["path"]
            if not path.exists():
                logger.error(
                    "Model file missing",
                    key=key,
                    path=str(path),
                )
                all_ok = False
                continue
            ok = verify_file_hash(path, info.get("sha256", ""))
            if not ok:
                all_ok = False

        if all_ok:
            logger.info(
                "Model integrity verified",
                symbol=symbol,
                timeframe=timeframe,
            )
        return all_ok

    def get_entry(
        self, symbol: str, timeframe: str
    ) -> dict[str, Any] | None:
        """Return the registry entry for *symbol* / *timeframe*, or None."""
        return self._load_registry().get(symbol, {}).get(timeframe)

    def list_symbols(self) -> list[str]:
        """Return all symbols that have entries in the registry."""
        return list(self._load_registry().keys())

    def all_entries(self) -> dict[str, dict[str, Any]]:
        """Return the full registry dict."""
        return self._load_registry()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _archive_version(
        self, symbol: str, timeframe: str, meta: dict[str, Any]
    ) -> None:
        """
        Move all files from *meta* into ``archive/<SYMBOL>/`` with a version
        suffix, then prune old archives to keep at most _KEEP_VERSIONS.  (K2)
        """
        version = meta.get("version", 0)
        arch_sym_dir = self._archive_dir / symbol
        try:
            arch_sym_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

        for _key, info in meta.get("files", {}).items():
            src = self.export_dir / info["path"]
            if src.exists():
                stem = src.stem
                dst = arch_sym_dir / f"{stem}_v{version}{src.suffix}"
                try:
                    src.rename(dst)
                except OSError as exc:
                    logger.warning(
                        "Could not archive model file",
                        src=str(src),
                        dst=str(dst),
                        error=str(exc),
                    )

        # Archive the metadata JSON too
        meta_src = self.export_dir / symbol / f"{symbol}_{timeframe}_meta.json"
        if meta_src.exists():
            meta_dst = arch_sym_dir / f"{symbol}_{timeframe}_v{version}_meta.json"
            try:
                meta_src.rename(meta_dst)
            except OSError:
                pass

        self._prune_archive(symbol, timeframe)

    def _prune_archive(self, symbol: str, timeframe: str) -> None:
        """Remove oldest archived versions so only _KEEP_VERSIONS remain."""
        arch_sym_dir = self._archive_dir / symbol
        if not arch_sym_dir.exists():
            return

        # Collect versioned metadata files for this symbol+TF
        pattern = f"{symbol}_{timeframe}_v*_meta.json"
        meta_files = sorted(
            arch_sym_dir.glob(pattern),
            key=lambda p: int(
                p.name.split("_v")[1].split("_")[0]
                if "_v" in p.name else "0"
            ),
        )
        # Keep only the most recent _KEEP_VERSIONS
        to_remove = meta_files[: max(0, len(meta_files) - _KEEP_VERSIONS)]
        for meta_path in to_remove:
            # Parse version from filename
            try:
                v = int(meta_path.name.split("_v")[1].split("_")[0])
            except (IndexError, ValueError):
                continue
            # Remove all artefacts with that version suffix
            for p in arch_sym_dir.glob(f"*_v{v}*"):
                try:
                    p.unlink()
                    logger.debug("Pruned old model archive", path=str(p))
                except OSError:
                    pass

    def _load_registry(self) -> dict[str, Any]:
        if not self._registry_path.exists():
            return {}
        try:
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Registry file unreadable, starting fresh",
                error=str(exc),
            )
            return {}

    def _save_registry(self, registry: dict[str, Any]) -> None:
        try:
            self._registry_path.write_text(
                json.dumps(registry, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.error(
                "Failed to save registry",
                error=str(exc),
            )
