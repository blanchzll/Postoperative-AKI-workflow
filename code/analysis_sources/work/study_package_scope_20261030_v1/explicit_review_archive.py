"""Allowlist-based archive I/O for the pending full-study review packages.

This module verifies file/ZIP integrity and content-class separation. It does
not approve research conclusions, decide shareability, publish files, execute
archive contents, or write PACKAGE_ACCEPTANCE. Composition and numerical replay
are separate requirements. Imports use the Python standard library only.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import stat
import zipfile

MANIFEST = "MANIFEST_SHA256.json"
SCHEMA = "AKI_EXPLICIT_REVIEW_ARCHIVE_V1"
CHUNK = 1024 * 1024
CLASSIFICATIONS = {"review_shareable", "private_only", "historical_private"}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()

def normalized_name(name):
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or ":" in name:
        raise ValueError("Invalid archive member name")
    p = PurePosixPath(name)
    if p.is_absolute() or str(p) != name or any(x in {"", ".", ".."} for x in p.parts):
        raise ValueError("Archive member is not a canonical relative path")
    if p.name == "" or (p.parts and p.parts[0].casefold() == MANIFEST.casefold()):
        raise ValueError("Reserved or empty archive member")
    return name

def validate_names(names):
    canonical = [normalized_name(n) for n in names]
    if len(canonical) != len(set(canonical)) or len(canonical) != len({n.casefold() for n in canonical}):
        raise ValueError("Duplicate or case-colliding archive members")
    all_names = {n.casefold() for n in canonical}
    for name in canonical:
        for parent in PurePosixPath(name).parents:
            if str(parent).casefold() in all_names:
                raise ValueError("Archive file/directory prefix collision")
    return canonical

def validate_selection(selection, archive_kind):
    if archive_kind not in {"shared_review", "private_review"} or not isinstance(selection, list) or not selection:
        raise ValueError("Explicit nonempty archive selection and kind required")
    names = validate_names([v["archive_path"] for v in selection])
    selected = {}
    for item, name in zip(selection, names):
        role = item.get("classification")
        if role not in CLASSIFICATIONS:
            raise ValueError("Unknown or unreviewed content classification")
        if archive_kind == "shared_review" and role != "review_shareable":
            raise ValueError("Private content prohibited in shared archive")
        h = item.get("sha256")
        if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError("Exact SHA256 required for every payload")
        size = item.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("Exact nonnegative byte length required")
        selected[name] = {"sha256": h, "bytes": size, "classification": role}
    return selected

def create_archive(source_root, allowed_physical_roots, selection, target, archive_kind,
                   selection_evidence_sha256):
    root = Path(source_root)
    allowed = tuple(Path(p).resolve(strict=True) for p in allowed_physical_roots)
    if not allowed or not all(p.is_dir() for p in allowed):
        raise ValueError("Explicit existing physical source roots required")
    payload = validate_selection(selection, archive_kind)
    if not isinstance(selection_evidence_sha256, str) or len(selection_evidence_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in selection_evidence_sha256
    ):
        raise ValueError("Content-selection evidence SHA required")
    target = Path(target)
    partial = target.with_name(target.name + ".partial")
    if target.exists() or target.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError("Existing archive or partial requires inspection")
    sources = {}
    for item in selection:
        relative = normalized_name(item["source_relative"])
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError("Payload must be a regular file, not a symlink")
        physical = source.resolve(strict=True)
        if not any(physical.is_relative_to(p) for p in allowed):
            raise ValueError("Payload is outside authorized physical project roots")
        if source.stat().st_size != item["bytes"] or digest(source) != item["sha256"]:
            raise ValueError("Source changed since explicit content selection")
        sources[item["archive_path"]] = source
    manifest = {
        "schema": SCHEMA, "archive_kind": archive_kind,
        "selection_evidence_sha256": selection_evidence_sha256,
        "files": payload,
        "scope": "Byte integrity and declared content-class separation only; not scientific or public-release acceptance",
    }
    metadata = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if len(metadata) > 8 * CHUNK:
        raise ValueError("Manifest exceeds bounded metadata limit")
    with zipfile.ZipFile(partial, "x", zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for name in sorted(payload):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            h = hashlib.sha256(); size = 0
            with sources[name].open("rb") as src, archive.open(info, "w", force_zip64=True) as dst:
                for block in iter(lambda: src.read(CHUNK), b""):
                    dst.write(block); h.update(block); size += len(block)
            if h.hexdigest() != payload[name]["sha256"] or size != payload[name]["bytes"]:
                raise ValueError("Payload changed during archive creation; partial retained")
        info = zipfile.ZipInfo(MANIFEST); info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o600) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, metadata)
    os.chmod(partial, 0o600)
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError("New archive CRC check failed; partial retained")
    # Exclusive publication on the same filesystem. Never replace old archives.
    os.link(partial, target)
    partial.unlink()
    return {
        "path": str(target), "bytes": target.stat().st_size, "sha256": digest(target),
        "archive_kind": archive_kind, "payload_members": len(payload),
        "payload_uncompressed_bytes": sum(v["bytes"] for v in payload.values()),
        "manifest_sha256": hashlib.sha256(metadata).hexdigest(),
        "selection_evidence_sha256": selection_evidence_sha256,
        "creation_CRC_pass": True, "not_package_acceptance": True,
    }

def inspect_manifest(archive, expected_kind, expected_selection_sha256, expected_payload_bytes):
    infos = archive.infolist()
    names = [i.filename for i in infos]
    if len(names) != len(set(names)) or names.count(MANIFEST) != 1:
        raise ValueError("Missing/duplicate ZIP names or manifest")
    other_names = [n for n in names if n != MANIFEST]
    validate_names(other_names)
    for info in infos:
        file_type = stat.S_IFMT(info.external_attr >> 16)
        if info.is_dir() or file_type not in {0, stat.S_IFREG} or info.flag_bits & 1:
            raise ValueError("Directory, link, special, or encrypted ZIP member rejected")
    mi = archive.getinfo(MANIFEST)
    if mi.file_size > 8 * CHUNK:
        raise ValueError("Oversized archive manifest")
    metadata = archive.read(MANIFEST)
    # Object-pair parsing rejects duplicate JSON keys rather than silently
    # replacing an earlier manifest entry with a later one.
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key in archive manifest")
            result[key] = value
        return result
    manifest = json.loads(metadata, object_pairs_hook=unique_pairs)
    if manifest["schema"] != SCHEMA or manifest["archive_kind"] != expected_kind:
        raise ValueError("Archive schema/kind mismatch")
    if manifest["selection_evidence_sha256"] != expected_selection_sha256:
        raise ValueError("Content-selection evidence mismatch")
    files = manifest["files"]
    if not isinstance(files, dict):
        raise ValueError("Manifest file map missing")
    validated = validate_selection([
        dict(archive_path=name, **item) for name, item in files.items()
    ], expected_kind)
    if set(validated) != set(other_names):
        raise ValueError("ZIP members differ from exact allowlist")
    total = sum(item["bytes"] for item in validated.values())
    if total != expected_payload_bytes:
        raise ValueError("Expected uncompressed payload size mismatch")
    for name, item in validated.items():
        if archive.getinfo(name).file_size != item["bytes"]:
            raise ValueError("ZIP member size differs from selected source")
    return manifest, metadata

def extract_verified(archive_path, destination, expected_archive_sha256, expected_kind,
                     expected_selection_sha256, expected_payload_bytes):
    archive_path = Path(archive_path)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Extraction destination must be new")
    if not archive_path.is_file() or digest(archive_path) != expected_archive_sha256:
        raise ValueError("Archive SHA256 mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        manifest, metadata = inspect_manifest(archive, expected_kind, expected_selection_sha256,
                                              expected_payload_bytes)
        destination.mkdir(mode=0o700)
        count = 0
        for name, expected in manifest["files"].items():
            out = destination / name
            out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            h = hashlib.sha256(); size = 0
            with archive.open(name) as src, out.open("xb") as dst:
                for block in iter(lambda: src.read(CHUNK), b""):
                    dst.write(block); h.update(block); size += len(block)
                dst.flush(); os.fsync(dst.fileno())
            os.chmod(out, 0o600)
            if size != expected["bytes"] or h.hexdigest() != expected["sha256"]:
                raise ValueError("Extracted payload hash mismatch; owned destination retained")
            count += 1
        with (destination/MANIFEST).open("xb") as handle:
            handle.write(metadata); handle.flush(); os.fsync(handle.fileno())
        os.chmod(destination/MANIFEST, 0o600)
        if archive.testzip() is not None:
            raise ValueError("Archive CRC check failed after extraction")
    return {
        "payload_members": count, "payload_uncompressed_bytes": expected_payload_bytes,
        "archive_sha256": expected_archive_sha256,
        "exact_allowlist_members": True, "every_payload_SHA256_pass": True,
        "CRC_pass": True, "destination": str(destination),
        "executed_archive_content": False, "not_package_acceptance": True,
    }
