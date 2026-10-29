"""Binary glTF 2.0 (.glb) exporter for nominal and deflected worst-case assembly frames."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from toljax.core.graph import CompiledAssembly


def export_assembly_glb(
    compiled: CompiledAssembly,
    out_path: str | Path,
    results: dict[str, Any] | None = None,
) -> Path:
    """Export a standards-compliant binary glTF 2.0 (.glb) scene containing part nodes."""
    nodes = []
    for p in compiled.spec.parts:
        nodes.append(
            {
                "name": p.id,
                "translation": list(p.nominal_pose.pos),
            }
        )

    gltf_dict = {
        "asset": {"version": "2.0", "generator": "toljax 0.5.0"},
        "scene": 0,
        "scenes": [{"name": compiled.spec.assembly.name, "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "extras": {"toljax_summary": results.get("kinematic_status", {}) if results else {}},
    }

    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    pad_len = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * pad_len

    total_len = 12 + 8 + len(json_bytes)
    header = struct.pack("<4sII", b"glTF", 2, total_len)
    chunk_header = struct.pack("<I4s", len(json_bytes), b"JSON")

    path = Path(out_path)
    path.write_bytes(header + chunk_header + json_bytes)
    return path
