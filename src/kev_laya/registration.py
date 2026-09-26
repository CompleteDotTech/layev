"""Explicit registration changes an administrator registry, never the Overwatch cache."""
from pathlib import Path
from .io import atomic_json
from .schema import strict_loads
from .telemetry_contract import decode_snapshot, identifier


def register(registry: Path, snapshot: Path, source_id: str):
    identifier(source_id)
    snapshot = snapshot.resolve()
    with snapshot.open("rb") as stream:
        decode_snapshot(stream.read(262145))
    current = strict_loads(registry.read_bytes()) if registry.exists() else {"schema_version": 1, "sources": []}
    if current.get("schema_version") != 1:
        raise ValueError("unsupported registry version")
    source = {"id": source_id, "transport": "local", "path": str(snapshot)}
    matches = [s for s in current["sources"] if s["id"] == source_id]
    if matches and matches != [source]:
        raise ValueError("source ID already names a different registration")
    if not matches:
        current["sources"].append(source)
    atomic_json(registry, current)
    return current
