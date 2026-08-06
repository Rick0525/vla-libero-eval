#!/usr/bin/env python
"""No-policy physics probe: does the task5 target bowl stay seated on the
ramekin after set_init_state + settle, under the CURRENT mujoco version?

Mirrors the official eval's settle phase (set_init_state, then let physics
run before the policy acts): for every official init state we set the state,
step the raw sim for --settle-steps physics steps (dt~=0.002s), and measure
the target bowl's xy drift relative to the ramekin, its z drop, and its tilt.

No policy, no rendering use beyond env construction. Run once per mujoco
version (driver swaps the wheel) and compare the JSON outputs.

Example:
  python scripts/mj_settle_probe.py --out /tmp/probe_381.json
"""

import argparse
import json

import numpy as np

from grasp_offset_analysis import body_xy_z, build_sim, find_bodies, set_state, target_bowl


def tilt_deg(sim, body: str) -> float:
    m = np.asarray(sim.data.get_body_xmat(body)).reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(m[2, 2], -1.0, 1.0))))


def rel(sim, bowl: str, ramekin: str):
    bxy, bz = body_xy_z(sim, bowl)
    rxy, rz = body_xy_z(sim, ramekin)
    return float(np.linalg.norm(np.asarray(bxy) - np.asarray(rxy))), float(bz - rz)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, default=5)
    p.add_argument("--settle-steps", type=int, default=250)
    p.add_argument("--off-cm", type=float, default=1.0, help="xy drift (cm) beyond which the bowl counts as slid off")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import mujoco

    le, sim = build_sim(args.suite, args.task_id)
    bowls, plate, ramekin = find_bodies(sim)
    checkpoints = sorted({0, 25, 50, 125, args.settle_steps})

    rows = []
    for i, st in enumerate(le._init_states):
        set_state(sim, st)
        tb = target_bowl(sim, bowls, ramekin)
        series = {}
        d0, dz0 = rel(sim, tb, ramekin)
        series[0] = (d0, dz0, tilt_deg(sim, tb))
        done = 0
        for t in checkpoints[1:]:
            for _ in range(t - done):
                sim.step()
            done = t
            d, dz = rel(sim, tb, ramekin)
            series[t] = (d, dz, tilt_deg(sim, tb))
        df, dzf, tf = series[args.settle_steps]
        rows.append(
            {
                "init": i,
                "bowl": tb,
                "xy_cm_t0": round(d0 * 100, 2),
                "xy_cm_final": round(df * 100, 2),
                "dz_cm_final": round(dzf * 100, 2),
                "tilt_deg_final": round(tf, 1),
                "slid_off": df * 100 > args.off_cm,
                "series": {str(k): [round(v[0] * 100, 2), round(v[1] * 100, 2), round(v[2], 1)] for k, v in series.items()},
            }
        )

    drifts = [r["xy_cm_final"] for r in rows]
    tilts = [r["tilt_deg_final"] for r in rows]
    summary = {
        "mujoco": mujoco.__version__,
        "suite": args.suite,
        "task_id": args.task_id,
        "settle_steps": args.settle_steps,
        "n_inits": len(rows),
        "n_slid_off": sum(r["slid_off"] for r in rows),
        "xy_cm_mean": round(float(np.mean(drifts)), 2),
        "xy_cm_max": round(float(np.max(drifts)), 2),
        "tilt_deg_mean": round(float(np.mean(tilts)), 1),
        "tilt_deg_max": round(float(np.max(tilts)), 1),
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
