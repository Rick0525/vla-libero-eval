#!/usr/bin/env python
"""Exp D instrument: audit the grasp-side distribution of raw LIBERO demos.

Why: task5 attribution settled on the control layer (grasp-geometry prior;
see Exp T/G) — the fix lives on the data side. Before choosing a data
intervention we audit the training distribution itself: for each raw demo,
which side of the bowl rim did the human grasp, from which layout, landing
where. Raw hdf5 demos carry per-step MuJoCo `states`, so the Exp G / bowl-
tilt calibers apply unchanged (states pushed into the sim, no policy, no
rendering, zero GPU).

Also marks each demo as kept/dropped by the OpenVLA regeneration pipeline
(`regenerate_libero_dataset.py`: no-op frames removed, then the whole demo
dropped unless its open-loop replay re-succeeds). Membership is recovered by
fingerprint: raw actions filtered with OpenVLA's no-op rule must equal one
lerobot episode's action sequence.

Example:
  python scripts/demo_grasp_audit.py --task-id 5 \
      --hdf5 /mnt/hdd16t/rick/vla_lab/datasets/LIBERO-raw/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5 \
      --out "$VLA_EVAL_OUTPUT_DIR/demo_audit_task5.json"
"""

import argparse
import json

import h5py
import numpy as np

from bowl_tilt_analysis import tilt_deg
from grasp_offset_analysis import (
    GRASP_LIFT_M,
    GRASP_NEAR_M,
    body_xy_z,
    build_sim,
    find_bodies,
    set_state,
    target_bowl,
)

NOOP_NORM = 1e-4  # OpenVLA regenerate_libero_dataset.py threshold


def openvla_noop_filter(actions: np.ndarray) -> np.ndarray:
    """Keep-mask under OpenVLA's rule: near-zero twist AND gripper same as prev."""
    keep = np.ones(len(actions), dtype=bool)
    prev_grip = None
    for t, a in enumerate(actions):
        small = np.linalg.norm(a[:-1]) < NOOP_NORM
        keep[t] = not (small and (prev_grip is None or a[-1] == prev_grip))
        prev_grip = a[-1]
    return keep


def measure_demo(sim, bowls, plate, ramekin, states: np.ndarray) -> dict:
    set_state(sim, states[0])
    tb = target_bowl(sim, bowls, ramekin)
    bxy0, z0 = body_xy_z(sim, tb)
    pxy0, _ = body_xy_z(sim, plate)

    lifted, near, tilts, bowl_minus_eef = [], [], [], []
    for st in states:
        set_state(sim, st)
        bxy, bz = body_xy_z(sim, tb)
        eef = sim.data.get_site_xpos("gripper0_grip_site")
        lifted.append(bz - z0 > GRASP_LIFT_M)
        near.append(float(np.linalg.norm(eef - np.array([*bxy, bz]))) < GRASP_NEAR_M)
        tilts.append(tilt_deg(sim.data.get_body_xmat(tb)))
        bowl_minus_eef.append([float(bxy[0] - eef[0]), float(bxy[1] - eef[1])])

    # Grasp events: contiguous lifted segments whose start also satisfies the
    # proximity test. >1 event = the bowl came back down and was re-grasped.
    events = []
    t = 0
    while t < len(states):
        if lifted[t] and (t == 0 or not lifted[t - 1]):
            seg_end = t
            while seg_end < len(states) and lifted[seg_end]:
                seg_end += 1
            grasp_t = next((k for k in range(t, seg_end) if near[k]), None)
            if grasp_t is not None:
                events.append(grasp_t)
            t = seg_end
        else:
            t += 1

    set_state(sim, states[-1])
    bxy, bz = body_xy_z(sim, tb)
    pxy, pz = body_xy_z(sim, plate)

    final_grasp = events[-1] if events else None
    carry = tilts[final_grasp:] if final_grasp is not None else []
    return {
        "layout_bowl_xy": [round(float(v), 4) for v in bxy0],
        "layout_bowl_minus_plate_xy": [round(float(a - b), 4) for a, b in zip(bxy0, pxy0)],
        "n_grasp_events": len(events),
        "first_grasp_step": events[0] if events else None,
        "final_grasp_step": final_grasp,
        "grasp_bowl_minus_eef_xy": (
            [round(v, 4) for v in bowl_minus_eef[final_grasp]] if final_grasp is not None else None
        ),
        "carry_mean_tilt_deg": round(float(np.mean(carry)), 1) if carry else None,
        "carry_max_tilt_deg": round(float(np.max(carry)), 1) if carry else None,
        "final_bowl_minus_plate_xy": [round(float(bxy[0] - pxy[0]), 4), round(float(bxy[1] - pxy[1]), 4)],
        "final_bowl_plate_dist": round(float(np.linalg.norm(bxy - pxy)), 4),
        "final_bowl_z_minus_plate_z": round(float(bz - pz), 4),
    }


def load_lerobot_episodes(repo_id: str, task_text: str):
    """Return {episode_index: action array} for episodes of the given task."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id)
    wanted = {
        ep["episode_index"]: (ep["dataset_from_index"], ep["dataset_to_index"])
        for ep in ds.meta.episodes
        if task_text in ep["tasks"]
    }
    # Column-wise extraction: row indexing would decode the image columns too.
    all_actions = np.asarray(ds.hf_dataset.select_columns(["action"])["action"])
    return {idx: all_actions[lo:hi] for idx, (lo, hi) in wanted.items()}


def match_episode(filtered_actions: np.ndarray, episodes: dict):
    """Match a no-op-filtered raw action sequence against lerobot episodes."""
    hits = []
    for ep_idx, acts in episodes.items():
        if abs(len(acts) - len(filtered_actions)) > 1:
            continue
        n = min(len(acts), len(filtered_actions))
        a, b = filtered_actions[:n], acts[:n]
        if np.max(np.abs(a[:, :6] - b[:, :6])) < 1e-3:
            grip_same = bool(np.max(np.abs(a[:, 6] - b[:, 6])) < 1e-3)
            hits.append((ep_idx, grip_same, len(acts) - len(filtered_actions)))
    return hits


def spearman(x, y):
    """Spearman rho with a permutation p-value (scipy-free fallback)."""
    try:
        from scipy.stats import spearmanr

        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except ImportError:
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        rho = float(np.corrcoef(rx, ry)[0, 1])
        rng = np.random.RandomState(0)
        perms = [np.corrcoef(rng.permutation(rx), ry)[0, 1] for _ in range(10000)]
        p = float(np.mean(np.abs(perms) >= abs(rho)))
        return rho, p


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--lerobot-repo", default="HuggingFaceVLA/libero", help="empty string skips membership matching")
    p.add_argument("--out")
    args = p.parse_args()

    le, sim = build_sim(args.suite, args.task_id)
    bowls, plate, ramekin = find_bodies(sim)

    f = h5py.File(args.hdf5, "r")
    task_text = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
    demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    print(f"task: {task_text} | demos: {len(demos)}")

    episodes = {}
    if args.lerobot_repo:
        episodes = load_lerobot_episodes(args.lerobot_repo, task_text)
        print(f"lerobot episodes for this task: {len(episodes)}")

    rows = []
    for k in demos:
        g = f["data"][k]
        states, actions = g["states"][:], g["actions"][:]
        row = {"demo": k, "n_steps": len(states), "success_flag": int(g["dones"][-1] or g["rewards"][-1])}
        row.update(measure_demo(sim, bowls, plate, ramekin, states))
        if episodes:
            filtered = actions[openvla_noop_filter(actions)]
            hits = match_episode(filtered, episodes)
            row["n_noop_frames"] = int(len(actions) - len(filtered))
            row["lerobot_match"] = hits[0][0] if len(hits) == 1 else None
            row["match_ambiguous"] = len(hits) > 1
            row["in_lerobot"] = len(hits) >= 1
        rows.append(row)
        print(row)

    # ---- summary ----------------------------------------------------------
    side_y = np.array([r["grasp_bowl_minus_eef_xy"][1] for r in rows if r["grasp_bowl_minus_eef_xy"]])
    layout_y = np.array([r["layout_bowl_minus_plate_xy"][1] for r in rows if r["grasp_bowl_minus_eef_xy"]])
    rho, pval = spearman(layout_y, side_y)

    suite_layout_y = []
    for st in le._init_states:
        set_state(sim, st)
        tb = target_bowl(sim, bowls, ramekin)
        suite_layout_y.append(round(float(body_xy_z(sim, tb)[0][1] - body_xy_z(sim, plate)[0][1]), 4))

    summary = {
        "n_demos": len(rows),
        "n_with_grasp": int(len(side_y)),
        "left_rim_y_lt_0": int(np.sum(side_y < 0)),
        "left_rim_y_lt_-5mm": int(np.sum(side_y < -0.005)),
        "center_band_abs_le_5mm": int(np.sum(np.abs(side_y) <= 0.005)),
        "n_regrasp_demos": int(sum(1 for r in rows if r["n_grasp_events"] > 1)),
        "spearman_layout_y_vs_side_y": [round(rho, 3), round(pval, 5)],
        "demo_layout_y_min_max": [round(float(layout_y.min()), 4), round(float(layout_y.max()), 4)],
        "eval_suite_layout_y_per_init": suite_layout_y,
        "demos_left_of_init3_layout": int(np.sum(layout_y <= suite_layout_y[3])),
    }
    if episodes:
        kept = [r for r in rows if r.get("in_lerobot")]
        dropped = [r for r in rows if not r.get("in_lerobot")]

        def grp(rs):
            sy = [r["grasp_bowl_minus_eef_xy"][1] for r in rs if r["grasp_bowl_minus_eef_xy"]]
            return {
                "n": len(rs),
                "mean_side_y": round(float(np.mean(sy)), 4) if sy else None,
                "left_rim": int(sum(1 for v in sy if v < 0)),
                "mean_final_dist": round(float(np.mean([r["final_bowl_plate_dist"] for r in rs])), 4),
                "mean_carry_tilt": round(
                    float(np.mean([r["carry_mean_tilt_deg"] for r in rs if r["carry_mean_tilt_deg"] is not None])), 1
                ),
                "mean_len": round(float(np.mean([r["n_steps"] for r in rs])), 1),
                "regrasps": int(sum(1 for r in rs if r["n_grasp_events"] > 1)),
            }

        summary["kept_vs_dropped"] = {"kept": grp(kept), "dropped": grp(dropped)}
        summary["n_matched"] = len(kept)
    print("\nSUMMARY:", json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as out:
            json.dump({"rows": rows, "summary": summary}, out, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
