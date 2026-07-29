#!/usr/bin/env python
"""Exp E-C instrument: add anchor-counterweight copies to the K=4 boost set.

Exp E verdict: the K=4 task5 upsample fixed the targeted init3 hole (0/13
all-time -> 4/10) but broke the guardrail (-14 episodes), with the damage
layout-locked on the sibling bowl-on-base tasks (t3 cookie box -9, t7 stove
-4) and full-video rulings showing target-anchor drift toward the ramekin.
Data-side premise confirmed: pre-boost the three anchors were balanced
(t3 1.57% / t5 1.63% / t7 1.74% of frames); the boost made ramekin ~1.8x
dominant (2.77% vs 1.55/1.72).

This build keeps the boost intact (single-variable difference vs the boost
arm) and restores anchor parity: duplicate t3 and t7 episodes once each, in
episode-index order, until each task's frame total reaches t5's. No
cherry-picking — the selection rule is deterministic and content-blind, so
the counterweight adds no new intra-task skew. meta/stats.json stays shared
verbatim with the baseline (same normalization across all arms).

Preregistered (7/29): guardrail-repair bet (t3+t7 combined diff vs ctrl
>= -5), benefit-retention bet (init3 probe >= 3/10 AND task5 >= 10/30), and
Rick's whack-a-mole bet (a NEW mole appears iff the other 7 tasks' combined
diff <= -4 or any single task <= -4; Rick: appears, Claude: does not).

Run on the server (zero GPU):
  python scripts/build_task5boost_bal_dataset.py
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
SRC = HOME / "local" / "libero_task5boost_k4"
OUT = HOME / "local" / "libero_task5boost_k4_bal"
T3 = "pick up the black bowl on the cookie box and place it on the plate"
T5 = "pick up the black bowl on the ramekin and place it on the plate"
T7 = "pick up the black bowl on the stove and place it on the plate"
ROWS_PER_FILE = 1064  # match the boost build's append-file granularity


def patched(table: pa.Table, col: str, values: pa.Array) -> pa.Table:
    return table.set_column(table.schema.get_field_index(col), col, values)


def task_frames(epi_rows: list[dict], task: str) -> int:
    return sum(r["length"] for r in epi_rows if task in r["tasks"])


def pick_counterweight(epi_rows: list[dict], task: str, target_frames: int) -> list[int]:
    """Episode-index order, one copy each, stop at first crossing of target."""
    picked, cum = [], 0
    for r in sorted((r for r in epi_rows if task in r["tasks"]), key=lambda r: r["episode_index"]):
        if cum >= target_frames:
            break
        picked.append(r["episode_index"])
        cum += r["length"]
    return picked


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

    epi_path = OUT / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    epi = pq.read_table(epi_path)
    epi_rows = epi.to_pylist()
    f5 = task_frames(epi_rows, T5)
    cw3 = pick_counterweight(epi_rows, T3, f5 - task_frames(epi_rows, T3))
    cw7 = pick_counterweight(epi_rows, T7, f5 - task_frames(epi_rows, T7))
    copy_eps = cw3 + cw7
    print(f"  counterweights: t3 {len(cw3)} eps, t7 {len(cw7)} eps (target {f5} frames/task)")

    print("step 2/4: locate counterweight episodes and append data files")
    ep_file = {}
    for f in data_files:
        u = set(pq.read_table(f, columns=["episode_index"])["episode_index"].to_pylist())
        for e in copy_eps:
            if e in u:
                ep_file.setdefault(e, f)
    assert sorted(ep_file) == sorted(copy_eps), "episodes not all located"

    next_file = int(data_files[-1].stem.split("-")[1]) + 1
    gidx, new_ep = n_src_frames, n_src_eps
    new_meta = []  # (new_ep, src_ep, from, to, data_file_index)
    cache, pending = {}, []

    def flush() -> None:
        nonlocal next_file, pending
        out_path = OUT / "data" / "chunk-000" / f"file-{next_file:03d}.parquet"
        pq.write_table(pa.concat_tables(pending), out_path, compression=src_codec)
        print(f"  wrote {out_path.name} ({sum(s.num_rows for s in pending)} rows)")
        next_file += 1
        pending = []

    for e in copy_eps:
        if ep_file[e] not in cache:
            cache[ep_file[e]] = pq.read_table(ep_file[e])
        sl = cache[ep_file[e]].filter(pc.equal(pc.field("episode_index"), e))
        n = sl.num_rows
        sl = patched(sl, "episode_index", pa.array([new_ep] * n, pa.int64()))
        sl = patched(sl, "index", pa.array(range(gidx, gidx + n), pa.int64()))
        new_meta.append((new_ep, e, gidx, gidx + n, next_file))
        gidx, new_ep = gidx + n, new_ep + 1
        pending.append(sl)
        if sum(s.num_rows for s in pending) >= ROWS_PER_FILE:
            flush()
    if pending:
        flush()

    print("step 3/4: append episodes metadata and update info.json")
    src_rows = {e: [r for r in epi_rows if r["episode_index"] == e][0] for e in copy_eps}
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

    out = LeRobotDataset("local/libero_task5boost_k4_bal", root=OUT)
    want_eps = n_src_eps + len(copy_eps)
    ok_counts = out.meta.total_episodes == want_eps and out.meta.total_frames == gidx
    print(f"  episodes {out.meta.total_episodes} (want {want_eps}) | frames {out.meta.total_frames} "
          f"(want {gidx}) -> {'OK' if ok_counts else 'MISMATCH'}")

    appended = [ep for ep in out.meta.episodes if ep["episode_index"] >= n_src_eps]
    n3 = sum(1 for ep in appended if T3 in ep["tasks"])
    n7 = sum(1 for ep in appended if T7 in ep["tasks"])
    ok_tasks = len(appended) == len(copy_eps) and n3 == len(cw3) and n7 == len(cw7)
    print(f"  appended episodes t3={n3} t7={n7}: {'OK' if ok_tasks else 'MISMATCH'}")

    new_rows = pq.read_table(epi_path).to_pylist()
    shares = {t: task_frames(new_rows, t) for t in (T3, T5, T7)}
    spread = max(shares.values()) - min(shares.values())
    ok_parity = spread <= 150  # within ~1.5 mean episodes of exact parity
    print(f"  anchor frame parity t3/t5/t7 = {shares[T3]}/{shares[T5]}/{shares[T7]} "
          f"(spread {spread}): {'OK' if ok_parity else 'MISMATCH'}")

    out_hf = out.hf_dataset.select_columns(["action", "episode_index", "index"])
    acts, epidx = np.asarray(out_hf["action"]), np.asarray(out_hf["episode_index"])
    ok_order = bool(np.array_equal(np.asarray(out_hf["index"]), np.arange(len(epidx))))
    print(f"  global row order == index column: {'OK' if ok_order else 'MISMATCH'}")

    ok_copy = np.array_equal(acts[epidx == n_src_eps], acts[epidx == cw3[0]]) and np.array_equal(
        acts[epidx == n_src_eps + len(cw3)], acts[epidx == cw7[0]])
    print(f"  first t3/t7 copies action-identical to sources: {'OK' if ok_copy else 'MISMATCH'}")

    frame = out[int(np.nonzero(epidx == n_src_eps)[0][0])]
    ok_frame = "observation.images.image" in frame and frame["observation.images.image"].shape[-2:] != ()
    print(f"  appended frame decodes: {'OK' if ok_frame else 'MISMATCH'}")

    src_ds = LeRobotDataset("local/libero_task5boost_k4", root=SRC)
    ok_src = src_ds.meta.total_episodes == n_src_eps and src_ds.meta.total_frames == n_src_frames
    print(f"  source (boost k4) untouched: {'OK' if ok_src else 'MISMATCH'}")

    ok_stats = json.load(open(OUT / "meta/stats.json")) == json.load(open(SRC / "meta/stats.json"))
    print(f"  stats identical to boost/baseline: {'OK' if ok_stats else 'MISMATCH'}")

    print("BUILD " + ("PASSED" if all(
        [ok_counts, ok_tasks, ok_parity, ok_order, ok_copy, ok_frame, ok_src, ok_stats]) else "FAILED"))


if __name__ == "__main__":
    main()
