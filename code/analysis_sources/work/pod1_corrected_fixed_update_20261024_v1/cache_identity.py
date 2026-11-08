"""Canonical identity helper; local route busy, Codex fallback under existing routing authorization."""
import hashlib,json
def canonical_sha256(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")).hexdigest()
