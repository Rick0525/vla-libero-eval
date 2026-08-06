#!/usr/bin/env python
"""Regenerate task5 init states as PRE-SETTLED states (and test seat stability).

The official task5 init states store the bowl floating ~11cm above the
ramekin and rely on physics settling at reset time; mujoco >=3.4.0 (box-box
distance bugfix) makes that drop land the bowl off-seat. This probe settles
the OBJECTS while PINNING the robot dofs (raw sim stepping has no controller,
so an unpinned arm sags and would contaminate the eval protocol), then stores
clean init rows: time=0, settled object qpos, original robot qpos, all-zero
qvel — byte-compatible with the official .pruned_init format.

  Phase A (--save, under a HEALTHY mujoco, e.g. 3.3.0):
    official init -> pinned settle -> store seated pre-settled states.
  Phase B (--load, under a DRIFTED mujoco, e.g. 3.8.1):
    load each seated state -> pinned settle -> measure additional drift.
    Zero extra drift = the seated equilibrium survives the new physics and
    the saved states are valid regenerated init states for it.

Example:
  python scripts/mj_reseat_probe.py --save settled_330.npz --out a.json   # under 3.3.0
  python scripts/mj_reseat_probe.py --load settled_330.npz \
      --save reseated_381.npz --out b.json                                # under 3.8.1
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


def robot_dof_indices(sim):
    """qpos/qvel index arrays for every robot-owned joint (arm + gripper)."""
    m = sim.model
    qpos_idx, qvel_idx = [], []
    for j in range(m.njnt):
        name = m.joint_id2name(j) or ""
        if "robot" not in name and "gripper" not in name:
            continue
        # robot joints are all hinge/slide: 1 qpos + 1 qvel each
        qpos_idx.append(int(m.jnt_qposadr[j]))
        qvel_idx.append(int(m.jnt_dofadr[j]))
    return np.asarray(qpos_idx), np.asarray(qvel_idx)


def pinned_settle(sim, steps: int, qpos_idx, qvel_idx):
    hold = np.asarray(sim.data.qpos)[qpos_idx].copy()
    for _ in range(steps):
        sim.step()
        sim.data.qpos[qpos_idx] = hold
        sim.data.qvel[qvel_idx] = 0.0
        sim.forward()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, default=5)
    p.add_argument("--settle-steps", type=int, default=250)
    p.add_argument("--load", help="npz of settled states to start from (Phase B); default: official init states")
    p.add_argument("--save", help="npz path to store the pre-settled init rows")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import mujoco

    le, sim = build_sim(args.suite, args.task_id)
    bowls, plate, ramekin = find_bodies(sim)
    qpos_idx, qvel_idx = robot_dof_indices(sim)
    starts = np.load(args.load)["states"] if args.load else np.asarray(le._init_states)

    nq = sim.model.nq
    rows, out_states = [], []
    for i, st in enumerate(starts):
        set_state(sim, st)
        tb = target_bowl(sim, bowls, ramekin)
        d0, dz0 = rel(sim, tb, ramekin)
        pinned_settle(sim, args.settle_steps, qpos_idx, qvel_idx)
        d1, dz1 = rel(sim, tb, ramekin)
        rows.append(
            {
                "init": i,
                "xy_cm_start": round(d0 * 100, 2),
                "xy_cm_final": round(d1 * 100, 2),
                "xy_cm_extra_drift": round((d1 - d0) * 100, 2),
                "dz_cm_start": round(dz0 * 100, 2),
                "dz_cm_final": round(dz1 * 100, 2),
                "tilt_deg_final": round(tilt_deg(sim, tb), 1),
                "robot_qpos_max_dev": round(float(np.max(np.abs(np.asarray(sim.data.qpos)[qpos_idx] - np.asarray(st)[1 : 1 + nq][qpos_idx]))), 6),
            }
        )
        # clean init row: time=0, settled qpos (robot pinned = original), zero qvel
        out_states.append(np.concatenate([[0.0], np.asarray(sim.data.qpos).copy(), np.zeros(sim.model.nv)]))

    if args.save:
        np.savez_compressed(args.save, states=np.stack(out_states).astype(np.float64))

    extra = [r["xy_cm_extra_drift"] for r in rows]
    summary = {
        "mujoco": mujoco.__version__,
        "phase": "B(reseat)" if args.load else "A(seat)",
        "n": len(rows),
        "n_robot_dofs_pinned": int(len(qpos_idx)),
        "xy_cm_start_mean": round(float(np.mean([r["xy_cm_start"] for r in rows])), 2),
        "xy_cm_final_mean": round(float(np.mean([r["xy_cm_final"] for r in rows])), 2),
        "extra_drift_cm_mean": round(float(np.mean(extra)), 2),
        "extra_drift_cm_max": round(float(np.max(extra)), 2),
        "tilt_final_mean": round(float(np.mean([r["tilt_deg_final"] for r in rows])), 1),
        "robot_qpos_max_dev": max(r["robot_qpos_max_dev"] for r in rows),
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
