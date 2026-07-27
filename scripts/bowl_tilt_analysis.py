#!/usr/bin/env python
"""Quantify Rick's visual grasp-quality criterion: bowl tilt during carry.

Observation (from full-video review of task5 sources): the marginal source
(init6 seed3100, own placement 3.0cm) carries the bowl visibly pitched
forward, while good sources (natural10 r002/r006) carry it level. Gripper-vs-
bowl xy offset does NOT separate these trajectories — tilt may be the missing
dimension of grasp quality. This script measures it from per-step MuJoCo
state dumps (attribution_probe.py --dump-states), no policy, no rendering.

Per rollout it reports: bowl tilt (deg, bowl z-axis vs world z) at t0 / max
during carry / at final step, the carry-window tilt series, plus two more
candidate grasp-quality scalars at the grasp moment: grip depth (eef z minus
bowl center z) and wrist axis tilt.

Example:
  python scripts/bowl_tilt_analysis.py --task-id 5 \
      --states-glob "$VLA_EVAL_OUTPUT_DIR/attr_init6_natural10/rollout_*/states.npz" \
      --out "$VLA_EVAL_OUTPUT_DIR/attr_init6_natural10/bowl_tilt.json"
"""

import argparse
import glob
import json

import numpy as np

from grasp_offset_analysis import (
    GRASP_LIFT_M,
    GRASP_NEAR_M,
    body_xy_z,
    build_sim,
    find_bodies,
    set_state,
    target_bowl,
)


def tilt_deg(rot3x3) -> float:
    # Angle between the body z-axis and world z.
    return float(np.degrees(np.arccos(np.clip(rot3x3[2, 2], -1.0, 1.0))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--states-glob", required=True)
    p.add_argument("--out")
    args = p.parse_args()

    le, sim = build_sim(args.suite, args.task_id)
    bowls, plate, ramekin = find_bodies(sim)
    report = []

    for path in sorted(glob.glob(args.states_glob)):
        states = np.load(path)["states"]
        set_state(sim, states[0])
        tb = target_bowl(sim, bowls, ramekin)
        _, z0 = body_xy_z(sim, tb)

        grasp_step = None
        grip_depth = wrist_tilt = None
        tilts = []
        for t, st in enumerate(states):
            set_state(sim, st)
            tilts.append(tilt_deg(sim.data.get_body_xmat(tb)))
            if grasp_step is None:
                bxy, bz = body_xy_z(sim, tb)
                eef = sim.data.get_site_xpos("gripper0_grip_site")
                if bz - z0 > GRASP_LIFT_M and np.linalg.norm(eef - np.array([*bxy, bz])) < GRASP_NEAR_M:
                    grasp_step = t
                    grip_depth = round(float(eef[2] - bz), 4)
                    wrist_tilt = round(tilt_deg(sim.data.get_site_xmat("gripper0_grip_site")), 1)

        carry = tilts[grasp_step:] if grasp_step is not None else []
        row = {
            "rollout": path.split("/")[-2],
            "grasp_step": grasp_step,
            "tilt_t0_deg": round(tilts[0], 1),
            "tilt_final_deg": round(tilts[-1], 1),
            "carry_max_tilt_deg": round(max(carry), 1) if carry else None,
            "carry_max_at_step": (grasp_step + int(np.argmax(carry))) if carry else None,
            "carry_mean_tilt_deg": round(float(np.mean(carry)), 1) if carry else None,
            "grip_depth_eef_minus_bowl_z": grip_depth,
            "wrist_axis_tilt_deg": wrist_tilt,
            "carry_tilt_series_deg": [round(x, 1) for x in carry],
        }
        report.append(row)
        print({k: v for k, v in row.items() if k != "carry_tilt_series_deg"})

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
