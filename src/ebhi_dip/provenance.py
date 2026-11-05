"""Portable source-text hashing and explicit legacy-to-package provenance verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ebhi_dip.paths import FROZEN_SPLIT_REL

SOURCE_HASH_SCHEME = "utf8-lf-v1"
PROVENANCE_MIGRATION_SCHEMA = "provenance-migration-v1"
PROVENANCE_MIGRATION_REL = Path("configs/provenance_migration.json")


def _reject_frozen_split_normalization(path: Path) -> None:
    if path.name == FROZEN_SPLIT_REL.name:
        raise ValueError(
            "Source-text newline normalization must never be applied to "
            f"'{FROZEN_SPLIT_REL.as_posix()}'; use verify_frozen_split or sha256_bytes."
        )


def sha256_bytes(path: str | Path) -> str:
    """Compute exact raw-byte SHA-256 hex digest of a file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing file for byte hashing: {file_path}")
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def normalize_source_bytes(raw_bytes: bytes) -> bytes:
    """Normalize UTF-8 source bytes under scheme utf8-lf-v1.

    Strips an optional leading UTF-8 BOM and normalizes CRLF/CR line endings to LF,
    while preserving all other whitespace and the presence or absence of a final newline.
    """
    text = raw_bytes.decode("utf-8")
    if text.startswith("\ufeff"):
        text = text[1:]
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.encode("utf-8")


def sha256_source_text(path: str | Path) -> str:
    """Compute portable source-text SHA-256 under scheme utf8-lf-v1."""
    file_path = Path(path)
    _reject_frozen_split_normalization(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing source file for text hashing: {file_path}")
    canonical_bytes = normalize_source_bytes(file_path.read_bytes())
    return hashlib.sha256(canonical_bytes).hexdigest()


def diagnose_legacy_source_hash(
    path: str | Path,
    expected_sha: str,
) -> dict[str, object]:
    """Diagnose whether a source file matches expected_sha via raw bytes, LF, or CRLF.

    Intended strictly for diagnosing legacy CRLF/LF source hash differences,
    not as a generic integrity bypass.
    """
    file_path = Path(path)
    _reject_frozen_split_normalization(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing source file for diagnosis: {file_path}")

    expected = expected_sha.strip().lower()
    raw = file_path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()

    lf_bytes = normalize_source_bytes(raw)
    crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
    lf_sha = hashlib.sha256(lf_bytes).hexdigest()
    crlf_sha = hashlib.sha256(crlf_bytes).hexdigest()

    return {
        "path": str(file_path),
        "scheme": SOURCE_HASH_SCHEME,
        "expected_sha256": expected,
        "actual_bytes_sha256": actual_sha,
        "lf_sha256": lf_sha,
        "crlf_sha256": crlf_sha,
        "exact_match": actual_sha == expected,
        "lf_match": lf_sha == expected,
        "crlf_match": crlf_sha == expected,
    }


def load_provenance_migration(project_root: str | Path) -> dict[str, object]:
    """Load and validate configs/provenance_migration.json."""
    root = Path(project_root).expanduser().resolve()
    migration_path = root / PROVENANCE_MIGRATION_REL
    if not migration_path.is_file():
        raise FileNotFoundError(f"Missing provenance migration record: {migration_path}")

    record = json.loads(migration_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != PROVENANCE_MIGRATION_SCHEMA:
        raise ValueError(
            f"Unsupported provenance migration schema: {record.get('schema_version')!r}"
        )
    if record.get("portable_source_scheme") != SOURCE_HASH_SCHEME:
        raise ValueError(
            f"Unsupported portable source scheme: {record.get('portable_source_scheme')!r}"
        )
    if not isinstance(record.get("legacy_helpers"), dict):
        raise ValueError("Provenance migration record missing 'legacy_helpers' mapping.")
    return record


def verify_helper_provenance(
    project_root: str | Path,
    role: str,
    recorded_sha256: str,
) -> dict[str, object]:
    """Verify historical helper provenance through the declared migration adapter.

    Accepts newline equivalence ONLY for explicitly declared historical helper digests,
    verifies the legacy helper if still present on disk, and enforces exact utf8-lf-v1
    integrity on the active migrated module.
    """
    root = Path(project_root).expanduser().resolve()
    migration = load_provenance_migration(root)
    helpers = migration["legacy_helpers"]
    if role not in helpers:
        raise ValueError(f"Unknown helper provenance role: {role!r}")

    entry = helpers[role]
    expected_recorded = str(entry["recorded_crlf_sha256"]).lower()
    expected_legacy_lf = str(entry["verified_utf8_lf_v1_sha256"]).lower()
    expected_migrated_lf = str(entry["migrated_module_utf8_lf_v1_sha256"]).lower()
    recorded = recorded_sha256.strip().lower()

    allowed_recorded = {expected_recorded, expected_migrated_lf}
    if recorded not in allowed_recorded:
        raise ValueError(
            f"Undeclared helper SHA-256 for '{role}': got {recorded}, "
            f"expected historical {expected_recorded} or migrated {expected_migrated_lf}."
        )

    legacy_rel = Path(entry["legacy_path"])
    legacy_path = root / legacy_rel
    legacy_status = "retired_to_package"
    legacy_diag: dict[str, object] | None = None
    if legacy_path.is_file():
        legacy_diag = diagnose_legacy_source_hash(legacy_path, expected_recorded)
        if (
            legacy_diag["lf_sha256"] != expected_legacy_lf
            or legacy_diag["crlf_sha256"] != expected_recorded
        ):
            raise ValueError(
                f"Legacy helper '{legacy_rel.as_posix()}' was modified on disk; "
                f"got utf8-lf-v1={legacy_diag['lf_sha256']}, expected {expected_legacy_lf}."
            )
        legacy_status = (
            "present_exact_crlf" if legacy_diag["exact_match"] else "present_verified_lf"
        )

    migrated_rel = Path(entry["migrated_module"])
    migrated_path = root / migrated_rel
    actual_migrated_lf = sha256_source_text(migrated_path)
    if actual_migrated_lf != expected_migrated_lf:
        raise ValueError(
            f"Active migrated module '{migrated_rel.as_posix()}' source mismatch: "
            f"got utf8-lf-v1={actual_migrated_lf}, expected {expected_migrated_lf}."
        )

    return {
        "role": role,
        "recorded_sha256": recorded,
        "legacy_path": legacy_rel.as_posix(),
        "legacy_status": legacy_status,
        "legacy_verified_utf8_lf_v1": expected_legacy_lf,
        "migrated_module": migrated_rel.as_posix(),
        "migrated_function": entry["migrated_function"],
        "migrated_utf8_lf_v1": actual_migrated_lf,
        "verified": True,
    }
