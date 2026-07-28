#!/usr/bin/env python
"""Exp E instrument: build the task5 deep-negative-band upsampled dataset (K=4).

Repair experiment for the task5 attribution verdict (control layer, grasp-side
prior; fix lives on the data side). The 9 kept demos whose initial bowl-minus-
plate y <= -1.25cm (the init3-like band where the policy collapses) are
physically replicated so each appears K=4 times in the training set. Everything
else is untouched: meta/stats.json is shared verbatim with the source, so both
retraining arms use the baseline normalization — the sampling mixture is the
only variable.

v2, hand-rolled parquet append. lerobot's official split/merge tools are
unusable here: in HuggingFaceVLA/libero the per-episode data/chunk_index &
data/file_index meta columns point at the wrong files for 1690/1693 episodes
(upstream conversion bug), while the global row order and dataset_from/to_index
ranges are fully consistent — training and every measurement we ran use only
the latter. This script therefore works purely with global row ranges:

  1. New root: hard-link all existing data parquets (33G, same filesystem),
     real-copy meta/.
  2. For each copy round, slice the 9 episodes' rows out of their actual data
     files (image bytes copied verbatim, no decode), patch episode_index and
     the global index column, write as new data/chunk-000/file-377+.parquet
     appended after the last existing file.
  3. Append 27 rows to meta/episodes (per-episode stats copied verbatim, file
     pointers set correctly for the new rows), update info.json totals.

Run on the server (zero GPU):
  python scripts/build_task5boost_dataset.py
"""

import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HOME = Path("/mnt/hdd16t/rick/vla_lab/datasets")
SRC = HOME / "HuggingFaceVLA" / "libero"
OUT = HOME / "local" / "libero_task5boost_k4"
# Deep-negative-band task5 demos (layout y <= -1.25cm), audit-matched lerobot
# episode indices — see results/attribution/demo_audit_task5.json (Exp D).
BOOST_EPS = [1265, 1396, 1404, 1433, 1469, 1525, 1684, 1685, 1692]
K = 4  # each boost episode appears K times in the output (1 original + K-1 copies)
TASK5 = "pick up the black bowl on the ramekin and place it on the plate"


def patched(table: pa.Table, col: str, values: pa.Array) -> pa.Table:
    return table.set_column(table.schema.get_field_index(col), col, values)


def main() -> None:
    assert not OUT.exists(), f"output already exists: {OUT}"

    print("step 1/4: lay out new root (hard-link data, copy meta)")
    (OUT / "data" / "chunk-000").mkdir(parents=True)
    data_files = sorted((SRC / "data" / "chunk-000").glob("file-*.parquet"))
    for f in data_files:
        os.link(f, OUT / "data" / "chunk-000" / f.name)
    shutil.copytree(SRC / "meta", OUT / "meta")

    info = json.load(open(SRC / "meta" / "info.json"))
    n_src_eps, n_src_frames = info["total_episodes"], info["total_frames"]
    src_codec = pq.ParquetFile(data_files[0]).metadata.row_group(0).column(0).compression

    print("step 2/4: locate boost episodes and append data files")
    ep_file = {}
    for f in data_files:
        u = set(pq.read_table(f, columns=["episode_index"])["episode_index"].to_pylist())
        for e in BOOST_EPS:
            if e in u:
                ep_file.setdefault(e, f)
    assert sorted(ep_file) == sorted(BOOST_EPS), f"episodes not all located: {sorted(ep_file)}"

    next_file = int(data_files[-1].stem.split("-")[1]) + 1
    gidx, new_ep = n_src_frames, n_src_eps
    new_meta = []  # (new_ep, src_ep, from, to, data_file_index)
    cache = {}
    for _ in range(1, K):
        slices = []
        for e in sorted(BOOST_EPS):
            if ep_file[e] not in cache:
                cache[ep_file[e]] = pq.read_table(ep_file[e])
            sl = cache[ep_file[e]].filter(pc.equal(pc.field("episode_index"), e))
            n = sl.num_rows
            sl = patched(sl, "episode_index", pa.array([new_ep] * n, pa.int64()))
            sl = patched(sl, "index", pa.array(range(gidx, gidx + n), pa.int64()))
            slices.append(sl)
            new_meta.append((new_ep, e, gidx, gidx + n, next_file))
            gidx, new_ep = gidx + n, new_ep + 1
        out_path = OUT / "data" / "chunk-000" / f"file-{next_file:03d}.parquet"
        pq.write_table(pa.concat_tables(slices), out_path, compression=src_codec)
        print(f"  wrote {out_path.name} ({sum(s.num_rows for s in slices)} rows)")
        next_file += 1

    print("step 3/4: append episodes metadata and update info.json")
    epi_path = OUT / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    epi = pq.read_table(epi_path)
    src_rows = {e: epi.filter(pc.equal(pc.field("episode_index"), e)).to_pylist()[0] for e in BOOST_EPS}
    rows = []
    for ne, e, lo, hi, fidx in new_meta:
        r = dict(src_rows[e])
        r.update({"episode_index": ne, "data/chunk_index": 0, "data/file_index": fidx,
                  "dataset_from_index": lo, "dataset_to_index": hi})
        rows.append(r)
    merged = pa.concat_tables([epi, pa.Table.from_pylist(rows, schema=epi.schema)])
    os.remove(epi_path)
    pq.write_table(merged, epi_path, compression=src_codec)

    info["total_episodes"] = new_ep
    info["total_frames"] = gidx
    info["splits"] = {"train": f"0:{new_ep}"}
    json.dump(info, open(OUT / "meta" / "info.json", "w"), indent=4)

    print("step 4/4: verify")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = LeRobotDataset("local/libero_task5boost_k4", root=OUT)
    want_eps = n_src_eps + (K - 1) * len(BOOST_EPS)
    ok_counts = out.meta.total_episodes == want_eps and out.meta.total_frames == gidx
    print(f"  episodes {out.meta.total_episodes} (want {want_eps}) | frames {out.meta.total_frames} "
          f"(want {gidx}) -> {'OK' if ok_counts else 'MISMATCH'}")

    appended = [ep for ep in out.meta.episodes if ep["episode_index"] >= n_src_eps]
    ok_tasks = len(appended) == (K - 1) * len(BOOST_EPS) and all(TASK5 in ep["tasks"] for ep in appended)
    print(f"  appended episodes all task5: {'OK' if ok_tasks else 'MISMATCH'}")

    out_hf = out.hf_dataset.select_columns(["action", "episode_index", "index"])
    acts, epidx = np.asarray(out_hf["action"]), np.asarray(out_hf["episode_index"])
    ok_order = bool(np.array_equal(np.asarray(out_hf["index"]), np.arange(len(epidx))))
    print(f"  global row order == index column: {'OK' if ok_order else 'MISMATCH'}")
    first_src = min(BOOST_EPS)
    ok_copy = np.array_equal(acts[epidx == n_src_eps], acts[epidx == first_src])
    print(f"  episode {n_src_eps} action-identical to source {first_src}: {'OK' if ok_copy else 'MISMATCH'}")

    frame = out[int(np.nonzero(epidx == n_src_eps)[0][0])]
    ok_frame = "observation.images.image" in frame and frame["observation.images.image"].shape[-2:] != ()
    print(f"  appended frame decodes: {'OK' if ok_frame else 'MISMATCH'}")

    src_ds = LeRobotDataset("HuggingFaceVLA/libero")
    ok_src = src_ds.meta.total_episodes == n_src_eps and src_ds.meta.total_frames == n_src_frames
    print(f"  source dataset untouched: {'OK' if ok_src else 'MISMATCH'}")

    ok_stats = json.load(open(OUT / "meta/stats.json")) == json.load(open(SRC / "meta/stats.json"))
    print(f"  stats identical to baseline: {'OK' if ok_stats else 'MISMATCH'}")

    print("BUILD " + ("PASSED" if all([ok_counts, ok_tasks, ok_order, ok_copy, ok_frame, ok_src, ok_stats]) else "FAILED"))


if __name__ == "__main__":
    main()
