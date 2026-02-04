from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diagram2code.datasets.validation import DatasetError


@dataclass(frozen=True)
class _SimPaths:
    split_path: Path
    images_dir: Path


_MERMAID_EDGE_RE = re.compile(r"entity(\d+)\s*[-=]+>\s*entity(\d+)")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_int(x: Any) -> int:
    return int(round(float(x)))


def _bbox_xyxy_to_xywh(x0: Any, y0: Any, x1: Any, y1: Any) -> list[int]:
    ix0, iy0, ix1, iy1 = _as_int(x0), _as_int(y0), _as_int(x1), _as_int(y1)
    w = max(0, ix1 - ix0)
    h = max(0, iy1 - iy0)
    return [ix0, iy0, w, h]


def _parse_simflowchart_edges(mermaid: str) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for m in _MERMAID_EDGE_RE.finditer(mermaid):
        edges.append((int(m.group(1)), int(m.group(2))))
    return edges


def _edges_from_meta_links(rec: dict[str, Any]) -> list[tuple[int, int]]:
    """
    Fallback: reconstruct edges from FlowLearn's meta.links when Mermaid parsing
    yields zero edges.

    meta: {
      "text": { "0": {"mermaid_entity_i": 0, ...}, ... },
      "links": { "1": {"start_text_i": "0", "end_text_i": "1", ...}, ... }
    }
    """
    meta = rec.get("meta")
    if not isinstance(meta, dict):
        return []

    text = meta.get("text")
    links = meta.get("links")
    if not isinstance(text, dict) or not isinstance(links, dict):
        return []

    # text index -> mermaid entity index
    text_i_to_entity: dict[str, int] = {}
    for text_i, tinfo in text.items():
        if not isinstance(tinfo, dict):
            continue
        ent = tinfo.get("mermaid_entity_i")
        if isinstance(ent, int):
            text_i_to_entity[str(text_i)] = ent

    edges: list[tuple[int, int]] = []
    for _lid, linfo in links.items():
        if not isinstance(linfo, dict):
            continue

        start_text_i = linfo.get("start_text_i")
        end_text_i = linfo.get("end_text_i")
        if not isinstance(start_text_i, str) or not isinstance(end_text_i, str):
            continue

        u = text_i_to_entity.get(start_text_i)
        v = text_i_to_entity.get(end_text_i)
        if u is None or v is None:
            continue

        edges.append((u, v))

    return edges


def _extract_simflowchart_nodes(rec: dict[str, Any]) -> list[dict[str, Any]]:
    meta = rec.get("meta")
    if not isinstance(meta, dict):
        raise DatasetError("SimFlowchart record missing 'meta' object")

    text = meta.get("text")
    if not isinstance(text, dict):
        raise DatasetError("SimFlowchart record missing 'meta.text' object")

    nodes_by_id: dict[str, dict[str, Any]] = {}

    for _k, v in text.items():
        if not isinstance(v, dict):
            continue

        ent = v.get("mermaid_entity_i")
        if ent is None:
            continue

        nid = str(int(ent))  # dataset contract: string IDs
        bbox = _bbox_xyxy_to_xywh(v.get("x0"), v.get("y0"), v.get("x1"), v.get("y1"))
        nodes_by_id[nid] = {"id": nid, "bbox": bbox}

    nodes = [nodes_by_id[k] for k in sorted(nodes_by_id, key=lambda s: int(s))]
    if not nodes:
        raise DatasetError("SimFlowchart record produced zero nodes from meta.text")
    return nodes


def _extract_simflowchart_edges(rec: dict[str, Any]) -> list[dict[str, Any]]:
    mermaid = rec.get("mermaid")
    if not isinstance(mermaid, str) or not mermaid.strip():
        raise DatasetError("SimFlowchart record missing 'mermaid' string")

    pairs = _parse_simflowchart_edges(mermaid)

    # Fallback: meta.links when Mermaid yields zero edges
    if not pairs:
        pairs = _edges_from_meta_links(rec)

    if not pairs:
        raise DatasetError(
            "SimFlowchart record produced zero edges (mermaid + meta.links fallback)"
        )

    # dataset contract: 'source'/'target' keys
    return [{"source": str(u), "target": str(v)} for (u, v) in pairs]


def _simflowchart_paths(flowlearn_root: Path, *, variant: str, split: str) -> _SimPaths:
    base = flowlearn_root / "SimFlowchart" / variant
    split_path = base / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    images_dir = flowlearn_root / "SimFlowchart" / "images" / f"{variant}_TextOCR" / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    return _SimPaths(split_path=split_path, images_dir=images_dir)


def _convert_simflowchart_variant(
    *,
    flowlearn_root: Path,
    variant: str,
    split: str,
    out: Path,
    limit: int | None,
    strict: bool,
) -> Path:
    paths = _simflowchart_paths(flowlearn_root, variant=variant, split=split)

    records = _read_json(paths.split_path)
    if not isinstance(records, list):
        raise DatasetError(f"Expected list JSON in {paths.split_path}")

    if limit is not None:
        records = records[: max(0, int(limit))]

    _ensure_dir(out)
    images_out = out / "images"
    graphs_out = out / "graphs"
    _ensure_dir(images_out)
    _ensure_dir(graphs_out)

    errors: list[str] = []
    sample_ids: list[str] = []

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            errors.append(f"[{idx}] record is not an object")
            if strict:
                break
            continue

        file_name = rec.get("file")
        if not isinstance(file_name, str) or not file_name.strip():
            errors.append(f"[{idx}] record missing 'file' string")
            if strict:
                break
            continue

        src_img = paths.images_dir / file_name
        if not src_img.exists():
            errors.append(f"[{idx}] missing image file: {src_img}")
            if strict:
                break
            continue

        sample_id = Path(file_name).stem
        dst_img = images_out / f"{sample_id}{src_img.suffix.lower()}"

        try:
            nodes = _extract_simflowchart_nodes(rec)
            edges = _extract_simflowchart_edges(rec)

            node_ids = {str(n["id"]) for n in nodes}
            for e in edges:
                if str(e["source"]) not in node_ids or str(e["target"]) not in node_ids:
                    raise DatasetError("Edge references missing node id")

            graph = {"nodes": nodes, "edges": edges}
            (graphs_out / f"{sample_id}.json").write_text(
                json.dumps(graph, indent=2),
                encoding="utf-8",
            )

            # Only copy image after graph succeeded (prevents orphan images)
            shutil.copy2(src_img, dst_img)

            sample_ids.append(sample_id)

        except Exception as e:
            errors.append(
                f"[{idx}] could not extract nodes/edges for {file_name}. "
                f"Record keys: {sorted(rec.keys())}. Error: {type(e).__name__}: {e}"
            )
            if strict:
                break
            continue

    if errors:
        msg = "FlowLearn conversion failed:\n" + "\n".join(errors[:30])
        if strict:
            raise RuntimeError(msg)
        print(msg)

    meta = {
        "schema_version": "1.0",
        "name": f"flowlearn_simflowchart_{variant}",
        "version": "0.1",
        "splits": {split: sample_ids},
        "extra": {
            "source": "jopan/FlowLearn",
            "subset": f"SimFlowchart/{variant}",
            "split_file": str(paths.split_path),
        },
    }
    (out / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def _convert_simflowchart_legacy_minimal(
    *,
    flowlearn_root: Path,
    split: str,
    out: Path,
) -> Path:
    """
    Legacy minimal format used by tests/test_convert_flowlearn_minimal.py:

      <root>/SimFlowchart/
        images/<img files>
        <split>.json   # list of records:
                       # {id, image, nodes:[{id,bbox}], edges:[{source,target}]}

    Convert to Phase-3 dataset contract.
    """
    sim_root = flowlearn_root / "SimFlowchart"
    split_path = sim_root / f"{split}.json"
    images_dir = sim_root / "images"

    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    records = _read_json(split_path)
    if not isinstance(records, list):
        raise DatasetError(f"Expected list JSON in {split_path}")

    _ensure_dir(out)
    images_out = out / "images"
    graphs_out = out / "graphs"
    _ensure_dir(images_out)
    _ensure_dir(graphs_out)

    sample_ids: list[str] = []

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise DatasetError(f"[{idx}] record is not an object")

        sid = rec.get("id")
        img = rec.get("image")
        nodes = rec.get("nodes")
        edges = rec.get("edges")

        if not isinstance(sid, str) or not sid:
            raise DatasetError(f"[{idx}] missing valid 'id'")
        if not isinstance(img, str) or not img:
            raise DatasetError(f"[{idx}] missing valid 'image'")
        if not isinstance(nodes, list):
            raise DatasetError(f"[{idx}] 'nodes' must be a list")
        if not isinstance(edges, list):
            raise DatasetError(f"[{idx}] 'edges' must be a list")

        src_img = images_dir / img
        if not src_img.exists():
            raise DatasetError(f"[{idx}] missing image file: {src_img}")

        dst_img = images_out / f"{sid}{src_img.suffix.lower()}"
        shutil.copy2(src_img, dst_img)

        norm_nodes: list[dict[str, Any]] = []
        for n in nodes:
            if not isinstance(n, dict):
                raise DatasetError(f"[{idx}] node must be an object")
            nid = n.get("id")
            bbox = n.get("bbox")
            if not isinstance(nid, str) or not nid:
                raise DatasetError(f"[{idx}] node missing valid 'id'")
            if not (
                isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox)
            ):
                raise DatasetError(f"[{idx}] node missing valid 'bbox' (4 ints)")
            norm_nodes.append({"id": nid, "bbox": bbox})

        norm_edges: list[dict[str, Any]] = []
        for e in edges:
            if not isinstance(e, dict):
                raise DatasetError(f"[{idx}] edge must be an object")
            s = e.get("source")
            t = e.get("target")
            if not isinstance(s, str) or not s:
                raise DatasetError(f"[{idx}] edge missing valid 'source'")
            if not isinstance(t, str) or not t:
                raise DatasetError(f"[{idx}] edge missing valid 'target'")
            norm_edges.append({"source": s, "target": t})

        graph = {"nodes": norm_nodes, "edges": norm_edges}
        (graphs_out / f"{sid}.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")

        sample_ids.append(sid)

    # ✅ IMPORTANT: match unit test expectation exactly
    meta = {
        "schema_version": "1.0",
        "name": "flowlearn-simflowchart",
        "version": "0.1",
        "splits": {split: sample_ids},
        "extra": {"subset": "SimFlowchart"},
    }
    (out / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def convert_flowlearn(
    *,
    flowlearn_root: Path,
    subset: str,
    split: str,
    out: Path | None = None,
    out_root: Path | None = None,
    limit: int | None = None,
    strict: bool = True,
) -> Path:
    """
    Convert FlowLearn -> diagram2code Phase-3 dataset format.

    Supported:
      - subset="SimFlowchart" (legacy minimal format for tests)
      - subset="SimFlowchart/char"
      - subset="SimFlowchart/word"

    Output:
      out/
        dataset.json
        images/<sample_id>.<ext>
        graphs/<sample_id>.json
    """
    flowlearn_root = Path(flowlearn_root)

    # Support old arg name used by tests
    if out is None and out_root is not None:
        out = out_root
    if out is None:
        raise TypeError("convert_flowlearn requires 'out' (or legacy alias 'out_root')")

    out = Path(out)

    if split not in {"train", "test", "all"}:
        raise ValueError("split must be one of: train, test, all")

    if subset == "SimFlowchart":
        return _convert_simflowchart_legacy_minimal(
            flowlearn_root=flowlearn_root,
            split=split,
            out=out,
        )

    if subset == "SimFlowchart/char":
        return _convert_simflowchart_variant(
            flowlearn_root=flowlearn_root,
            variant="char",
            split=split,
            out=out,
            limit=limit,
            strict=strict,
        )

    if subset == "SimFlowchart/word":
        return _convert_simflowchart_variant(
            flowlearn_root=flowlearn_root,
            variant="word",
            split=split,
            out=out,
            limit=limit,
            strict=strict,
        )

    raise ValueError(
        "Unsupported subset. Use: SimFlowchart, SimFlowchart/char, or SimFlowchart/word."
    )
