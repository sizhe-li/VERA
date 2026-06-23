"""Per-source episode discovery + path resolution.

A ``Source`` enumerates episodes for one data source (DROID, allegro_sim, mimicgen, pusht) and
resolves each episode's per-view video / flow / packed-NPZ / trajectory paths, plus its native fps.
This replaces BOTH original discovery paths (okto ``metadata_builder.py`` JSON cache and flow-planner
``droid_flow`` CSV metadata) with one cache that serves both consumers.

Phase 0 = contract + ``DroidSource`` skeleton + native-fps table. Phase 3 ports the actual walk/cache
from the two originals.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Per-source native fps (from the original combined_4env: DROID=30, others=10). Used by FrameSamplerConfig.
NATIVE_FPS: Dict[str, int] = {
    "droid": 30,
    "allegro_sim": 10,
    "allegro_real": 10,
    "mimicgen": 10,
    "robomimic": 10,  # packed mimicgen/robomimic episodes (cfg name="robomimic")
    "iiwa": 10,
    "pusht": 10,
}


@dataclass
class Episode:
    """One trajectory. ``paths`` holds per-view + packed/trajectory file locations."""

    episode_id: str
    source: str
    num_frames: int
    views: Sequence[str]
    paths: Dict[str, Any] = field(default_factory=dict)
    native_fps: int = 10
    # Free-form per-episode metadata read from the discovery CSV/index: caption
    # (priority caption->gemini_caption->original_caption, resolved by the dataset),
    # task_class, native per-view height/width. Consumed by the WAN video path
    # (_getitem_video) to build prompts / task_class. Empty for packed episodes.
    meta: Dict[str, Any] = field(default_factory=dict)


class Source(abc.ABC):
    """Enumerates + resolves episodes for one data source."""

    name: str

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)

    @abc.abstractmethod
    def list_episodes(self) -> List[Episode]:
        """Discover all episodes under ``data_root`` (cached)."""

    def native_fps(self) -> int:
        return NATIVE_FPS.get(self.name, 10)


class DroidSource(Source):
    """DROID raw (ext1/ext2/wrist) — discovery from the success-filtered, captioned
    multiview CSV that flow-planner's ``DroidFlowDataset`` consumes.

    The DROID metadata CSV is a WIDE, one-row-per-trajectory table (NO ``view``
    column): each row carries ``view_{ext1,ext2,wrist}_video_path`` columns, an
    anchor ``video_path`` (the ext1 view), ``fps`` (= flow-planner ``override_fps``,
    stamped into ``record["fps"]``), ``n_frames``, ``height``/``width``, ``success``
    and caption columns (``gemini_caption``/``original_caption``). flow-planner's
    ``_load_records`` keeps only rows that pass filtering; the ``_cleaned_`` CSV is
    already success-filtered to ``success/`` trajectories, but we additionally drop
    any ``success==False`` row (and any ``failure/`` path) so vera enumerates exactly
    the same episode set even if pointed at an unfiltered CSV.

    Accepts ``metadata_path`` (the subset cfg points at a CSV relative to or under
    ``data_root``) and ``concat_views`` (canonical left->right view order for the
    tiled WAN canvas). Populates ``Episode.meta`` with caption (priority
    ``caption``->``gemini_caption``->``original_caption``), ``task_class`` and native
    ``height``/``width``, mirroring flow-planner ``video_base.py`` lines 165-172 +
    229. Per-view frame counts can differ; the loader resolves the shortest at load
    time (``DroidViewLoader.effective_num_frames``) to match flow-planner's
    ``min_n_frames``."""

    name = "droid"
    CANONICAL_VIEWS = ("ext1", "ext2", "wrist")

    def __init__(
        self,
        data_root: str | Path,
        *,
        metadata_path: str | Path | None = None,
        concat_views: Sequence[str] | None = None,
    ):
        super().__init__(data_root)
        self._metadata_path = Path(metadata_path) if metadata_path else None
        self._concat_views = list(concat_views) if concat_views else None

    def _resolve_csv(self) -> Path:
        # combined_4env points metadata_path at a CSV under data_root (relative name);
        # also accept an absolute path or a bare glob fallback.
        if self._metadata_path is not None:
            cand = self._metadata_path
            if cand.is_file():
                return cand
            cand = self.data_root / self._metadata_path
            if cand.is_file():
                return cand
        csvs = sorted(self.data_root.glob("metadata_*.csv")) or sorted(
            self.data_root.glob("*.csv")
        )
        if not csvs:
            raise FileNotFoundError(
                f"No DROID metadata CSV under {self.data_root} (metadata_path="
                f"{self._metadata_path})"
            )
        return csvs[0]

    @staticmethod
    def _caption(row: Dict[str, Any]) -> str:
        for key in ("caption", "gemini_caption", "original_caption"):
            val = row.get(key)
            if val not in (None, "", "nan", "NaN"):
                return str(val)
        return ""

    @staticmethod
    def _truthy(val: Any) -> bool:
        # CSV ``success`` may be bool / "True" / "true" / 1.
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        return s in ("true", "1", "1.0", "yes")

    def list_episodes(self) -> List[Episode]:  # noqa: D401
        cached = getattr(self, "_episodes_cache", None)
        if cached is not None:
            return cached
        import csv

        views = self._concat_views or list(self.CANONICAL_VIEWS)
        view_cols = {v: f"view_{v}_video_path" for v in views}
        episodes: List[Episode] = []
        csv_path = self._resolve_csv()
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                # Success filter (flow-planner enumerates only success/ trajectories).
                if "success" in row and not self._truthy(row.get("success")):
                    continue
                anchor = row.get("video_path") or ""
                if "/failure/" in anchor or anchor.startswith("failure/"):
                    continue
                if not all(
                    row.get(view_cols[v]) and str(row.get(view_cols[v])) != "nan"
                    for v in views
                ):
                    continue  # skip rows missing a view
                episodes.append(
                    Episode(
                        episode_id=row.get("trajectory_id") or anchor,
                        source="droid",
                        num_frames=int(float(row["n_frames"])),
                        views=list(views),
                        paths={
                            "videos": {
                                v: str(self.data_root / row[view_cols[v]])
                                for v in views
                            }
                        },
                        native_fps=int(float(row.get("fps", 10) or 10)),
                        meta={
                            "caption": self._caption(row),
                            "task_class": str(row.get("task_class") or ""),
                            "native_height": int(float(row.get("height", 0) or 0)),
                            "native_width": int(float(row.get("width", 0) or 0)),
                            "video_path": anchor,
                        },
                    )
                )
        if not episodes:
            raise RuntimeError(
                f"DroidSource found no usable episodes in {csv_path} "
                "(after success/view filtering)."
            )
        self._episodes_cache = episodes
        return episodes


class LongFormatVideoSource(Source):
    """Discovery for the 'long' metadata CSV used by allegro_sim / allegro_real /
    mimicgen (one row per (trajectory, view): columns ``trajectory_id``, ``view``,
    ``video_path``, ``n_frames``, ``fps``). Rows are grouped by ``trajectory_id`` and
    the per-view mp4 paths collected into one :class:`Episode`. The same
    :class:`DroidViewLoader` decode path serves these (it iterates ``episode.views``).

    Subclasses set ``name`` (and optionally ``CSV_GLOB``). ``data_root`` is the dir
    holding both the metadata CSV and the relative video tree.
    """

    name = "long_video"
    CSV_GLOB = "*.csv"

    def __init__(
        self,
        data_root: str | Path,
        *,
        metadata_path: str | Path | None = None,
        concat_views: Sequence[str] | None = None,
    ):
        super().__init__(data_root)
        # Explicit metadata CSV (combined_4env subsets point at an absolute path that
        # is NOT under data_root). Falls back to CSV_GLOB under data_root.
        self._metadata_path = Path(metadata_path) if metadata_path else None
        # Canonical view order (the subset cfg's ``concat_views``). When set, episodes
        # are emitted with views in this exact order (so the tiled layout matches
        # flow-planner _load_video_concat's left->right view order). Rows whose view is
        # not in this set are ignored.
        self._concat_views = list(concat_views) if concat_views else None

    def _resolve_csv(self) -> Path:
        if self._metadata_path is not None and self._metadata_path.is_file():
            return self._metadata_path
        csvs = sorted(self.data_root.glob(self.CSV_GLOB)) or sorted(self.data_root.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No metadata CSV ({self.CSV_GLOB}) under {self.data_root} for source '{self.name}'"
            )
        return csvs[0]

    def _caption(self, row: Dict[str, Any]) -> str:
        # flow-planner video_base.py lines 165-171: caption priority.
        for key in ("caption", "gemini_caption", "original_caption"):
            val = row.get(key)
            if val not in (None, "", "nan"):
                return str(val)
        return ""

    def list_episodes(self) -> List[Episode]:  # noqa: D401
        cached = getattr(self, "_episodes_cache", None)
        if cached is not None:
            return cached
        import csv
        from collections import OrderedDict

        csv_path = self._resolve_csv()
        grouped: "OrderedDict[str, list]" = OrderedDict()
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                tid = row.get("trajectory_id") or row.get("video_path")
                grouped.setdefault(tid, []).append(row)

        episodes: List[Episode] = []
        for tid, rows in grouped.items():
            by_view = {r["view"]: r for r in rows}
            # Anchor row (the trajectory's CSV record). With the dualview mimicgen CSV
            # there is ONE row per trajectory (view=agentview_image_rgb) and the second
            # view is resolved from the filesystem convention, exactly like flow-planner
            # droid_flow._resolve_view_video_path. With a true per-view long CSV (e.g.
            # allegro) every concat view has its own row.
            r0 = rows[0]
            if self._concat_views is not None:
                order = list(self._concat_views)
            else:
                order = [r["view"] for r in rows]
            # flow-planner _resolve_view_video_path(record, view): prefer a
            # `view_{view}_video_path` column; else traj_dir/{view}.mp4 where
            # traj_dir = data_root / Path(record["video_path"]).parent.
            traj_dir = self.data_root / Path(r0["video_path"]).parent
            paths: Dict[str, str] = {}
            for v in order:
                row_v = by_view.get(v, r0)
                col = row_v.get(f"view_{v}_video_path")
                if col and str(col) != "nan":
                    paths[v] = str(self.data_root / col)
                elif v in by_view:
                    paths[v] = str(self.data_root / by_view[v]["video_path"])
                else:
                    paths[v] = str(traj_dir / f"{v}.mp4")
            num_frames = int(r0["n_frames"])  # anchor record's frame count
            fps = int(float(r0.get("fps", 10) or 10))
            episodes.append(
                Episode(
                    episode_id=tid,
                    source=self.name,
                    num_frames=num_frames,
                    views=order,
                    paths={"videos": paths},
                    native_fps=fps,
                    meta={
                        "caption": self._caption(r0),
                        "task_class": str(r0.get("task_class") or ""),
                        "native_height": int(float(r0.get("height", 0) or 0)),
                        "native_width": int(float(r0.get("width", 0) or 0)),
                        # The per-row video_path of the first view, used by the WAN
                        # output `video_path` key (flow-planner sets it from record).
                        "video_path": r0.get("video_path", ""),
                    },
                )
            )
        self._episodes_cache = episodes
        return episodes


class PackedSource(Source):
    """Discovery for okto's packed-NPZ format (one ``.npz`` per episode).

    Replaces okto ``metadata_builder.build_metadata`` for the packed path: enumerate
    episodes from ``index.json`` (a flat list of shard-relative ``.npz`` paths, written
    by the packer) and resolve each episode's packed metadata (num_frames, views,
    flow/trajectory entries) lazily on first access — so a 9000-episode root costs no
    I/O at construction time. The native-fps table (``NATIVE_FPS``) is keyed by
    ``self.name`` so any embodiment routed here (robomimic/mimicgen/iiwa/pusht) gets the
    right fps without hardcoding.

    Each :class:`Episode` carries ``paths={"packed_npz": <abs path>}`` and a ``packed``
    summary (rgb/flow/trajectory entry descriptors); the :class:`PackedViewLoader`
    consumes both. Mirrors okto's per-episode ``meta`` dict shape:
    ``{"paths": {"packed_npz": ...}, "views": [...], "num_frames": N, "packed": {...}}``.
    """

    name = "robomimic"

    def __init__(self, data_root, *, name: str | None = None, views=None):
        super().__init__(data_root)
        if name is not None:
            self.name = str(name)
        # Configured camera views (canonical order from the dataset cfg). When None,
        # every view present in the packed metadata is used (in packed order).
        self._configured_views = list(views) if views else None

    def _episode_paths(self) -> List[Path]:
        index_path = self.data_root / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"PackedSource expects an index.json under {self.data_root} "
                "(flat list of shard-relative .npz paths)."
            )
        import json

        index_data = json.loads(index_path.read_text())
        if isinstance(index_data, dict):
            # Rich index: {"episodes": [{"path": ...}, ...]}.
            episodes_list = index_data.get("episodes", [])
            rels = [ep["path"] for ep in episodes_list]
        elif isinstance(index_data, list):
            rels = list(index_data)
        else:
            raise ValueError(f"Unrecognized index.json format: {type(index_data)}")
        return [self.data_root / Path(rel) for rel in rels]

    def list_episodes(self) -> List[Episode]:  # noqa: D401
        cached = getattr(self, "_episodes_cache", None)
        if cached is not None:
            return cached
        npz_paths = self._episode_paths()
        if not npz_paths:
            raise RuntimeError(f"PackedSource found no episodes under {self.data_root}")
        fps = self.native_fps()
        episodes: List[Episode] = [
            Episode(
                episode_id=p.stem,
                source=self.name,
                num_frames=-1,  # resolved lazily by resolve_episode()
                views=list(self._configured_views) if self._configured_views else [],
                paths={"packed_npz": str(p)},
                native_fps=fps,
            )
            for p in npz_paths
        ]
        self._episodes_cache = episodes
        return episodes

    def resolve_episode(self, episode: Episode) -> Episode:
        """Populate ``num_frames`` / ``views`` / ``packed`` from packed metadata.

        Idempotent + cached on the Episode (``num_frames`` >= 0 means resolved). The
        view set is intersected with the configured cfg views (preserving cfg order)
        so an embodiment with extra packed views still trains on the requested set —
        matching okto ``_build_one_packed_episode``'s ``selected_views``.
        """
        if episode.num_frames >= 0:
            return episode
        from vera.datasets.core.packed import load_packed_metadata

        meta = load_packed_metadata(episode.paths["packed_npz"])
        rgb_entries = meta.get("rgb_entries") or {}
        flow_entries = meta.get("flow_entries") or {}
        traj_entries = meta.get("trajectory_entries") or {}
        available = list(rgb_entries.keys())
        if self._configured_views:
            views = [v for v in self._configured_views if v in available]
        else:
            views = available
        episode.views = views
        episode.num_frames = int(meta.get("num_frames", 0))
        episode.paths = dict(episode.paths)
        episode.paths["packed"] = {
            "rgb_entries": {v: rgb_entries.get(v, {}) for v in views},
            "flow_entries": {v: flow_entries.get(v, {}) for v in views},
            "trajectory_entries": traj_entries,
        }
        return episode


class AllegroSimSource(LongFormatVideoSource):
    name = "allegro_sim"
    CSV_GLOB = "allegro_apptainer*.csv"


class AllegroRealSource(LongFormatVideoSource):
    name = "allegro_real"
    CSV_GLOB = "allegro_real*.csv"


class MimicgenSource(LongFormatVideoSource):
    name = "mimicgen"
    CSV_GLOB = "mimicgen_*.csv"
