from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from diagram2code.datasets.validation import DatasetError


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    root: Path

    @property
    def metadata_path(self) -> Path:
        return self.root / "dataset.json"

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def graphs_dir(self) -> Path:
        return self.root / "graphs"


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    schema_version: str
    name: str = ""
    version: str = ""
    splits: dict[str, tuple[str, ...]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetSample:
    sample_id: str
    image_path: Path
    graph_path: Path
    split: str = "all"

    def load_graph_json(self) -> dict[str, Any]:
        raw = json.loads(self.graph_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Graph JSON must be an object: {self.graph_path}")
        return raw


@dataclass(frozen=True, slots=True)
class Dataset:
    root: Path
    metadata: DatasetMetadata
    _samples: tuple[DatasetSample, ...] = ()

    @classmethod
    def from_json_dict(
        cls, *, root: Path, data: dict[str, Any], validate_graphs: bool = True
    ) -> Dataset:
        if "schema_version" not in data:
            raise DatasetError("Missing schema_version in dataset.json")

        schema_version = str(data["schema_version"])
        # tests expect 1.x to be accepted, 9.9 to error
        if not (schema_version.startswith("1") or schema_version.startswith("1.")):
            raise DatasetError(f"Unsupported schema_version: {schema_version}")

        layout = DatasetLayout(root)

        # Normalize splits -> dict[str, tuple[str,...]]
        raw_splits = data.get("splits") or {}
        splits: dict[str, tuple[str, ...]] = {}
        if isinstance(raw_splits, dict):
            for k, v in raw_splits.items():
                if isinstance(v, list):
                    splits[str(k)] = tuple(str(x) for x in v)

        meta = DatasetMetadata(
            schema_version=schema_version,
            name=str(data.get("name", "")),
            version=str(data.get("version", "")),
            splits=splits,
            extra={
                k: v
                for k, v in data.items()
                if k not in {"schema_version", "name", "version", "splits"}
            },
        )

        # --- Build samples ---
        # Support both schemas:
        # A) "samples": [{"id":..., "image_path":..., ...}] (FlowLearn converted)
        # B) classic: infer ids from images/ + graphs/ and optionally use splits mapping
        samples: list[DatasetSample] = []

        if isinstance(data.get("samples"), list):
            for rec in data["samples"]:
                if not isinstance(rec, dict):
                    continue
                sid = str(rec.get("id", ""))
                img_rel = rec.get("image_path")
                if not sid or not img_rel:
                    continue
                img_path = (root / str(img_rel)).resolve()
                # graph is optional for some datasets; but most tests rely on graphs existing.
                graph_rel = rec.get("graph_path")
                graph_path = (
                    (root / str(graph_rel)).resolve()
                    if graph_rel
                    else (layout.graphs_dir / f"{sid}.json")
                )
                samples.append(
                    DatasetSample(
                        sample_id=sid, image_path=img_path, graph_path=graph_path, split="all"
                    )
                )
        else:
            # infer ids by matching images/* and graphs/*
            if not layout.images_dir.exists():
                raise DatasetError(f"Missing images/ directory: {layout.images_dir}")
            if not layout.graphs_dir.exists():
                raise DatasetError(f"Missing graphs/ directory: {layout.graphs_dir}")

            img_ids = {p.stem for p in layout.images_dir.iterdir() if p.is_file()}
            graph_ids = {
                p.stem for p in layout.graphs_dir.iterdir() if p.is_file() and p.suffix == ".json"
            }

            # mismatch should raise (tests expect this)
            if img_ids != graph_ids:
                missing_graph = sorted(img_ids - graph_ids)
                missing_img = sorted(graph_ids - img_ids)
                raise DatasetError(
                    "Image/graph id mismatch. "
                    f"Missing graphs for: {missing_graph[:5]} "
                    f"Missing images for: {missing_img[:5]}"
                )

            # validate split ids exist
            all_ids = img_ids
            for split_name, ids in splits.items():
                for sid in ids:
                    if sid not in all_ids:
                        raise DatasetError(f"Split '{split_name}' contains unknown id: {sid}")

            # invert split map -> id -> split
            id_to_split: dict[str, str] = {}
            for split_name, ids in splits.items():
                for sid in ids:
                    id_to_split[sid] = split_name

            for sid in sorted(all_ids):
                samples.append(
                    DatasetSample(
                        sample_id=sid,
                        image_path=layout.images_dir / f"{sid}.png"
                        if (layout.images_dir / f"{sid}.png").exists()
                        else next(
                            (
                                p
                                for p in layout.images_dir.iterdir()
                                if p.is_file() and p.stem == sid
                            ),
                            layout.images_dir / f"{sid}.png",
                        ),
                        graph_path=layout.graphs_dir / f"{sid}.json",
                        split=id_to_split.get(sid, "all"),
                    )
                )

        ds = cls(root=root, metadata=meta, _samples=tuple(samples))

        if validate_graphs:
            for s in ds:
                try:
                    g = s.load_graph_json()
                except Exception as e:
                    raise DatasetError(f"Invalid JSON: {s.graph_path}") from e

                if not isinstance(g.get("nodes"), list) or not isinstance(g.get("edges"), list):
                    raise DatasetError(f"Graph missing nodes/edges: {s.graph_path}")

                for n in g["nodes"]:
                    if not isinstance(n, dict) or not isinstance(n.get("id"), str) or not n["id"]:
                        raise DatasetError(f"Graph node missing valid 'id': {s.graph_path}")

                for e in g["edges"]:
                    if not isinstance(e, dict):
                        raise DatasetError(f"Graph edge missing valid 'source': {s.graph_path}")

                    if not isinstance(e.get("source"), str) or not e["source"]:
                        raise DatasetError(f"Graph edge missing valid 'source': {s.graph_path}")

                    if not isinstance(e.get("target"), str) or not e["target"]:
                        raise DatasetError(f"Graph edge missing valid 'target': {s.graph_path}")

        return ds

    def splits(self) -> tuple[str, ...]:
        return tuple(sorted(self.metadata.splits.keys()))

    def samples(self, split: str | None = None) -> Iterable[DatasetSample]:
        if split is None:
            return self._samples
        return tuple(s for s in self._samples if s.split == split)

    @property
    def items(self) -> tuple[DatasetSample, ...]:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)
