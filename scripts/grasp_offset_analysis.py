#!/usr/bin/env python
"""Offline geometry analysis for the grasp-offset hypothesis.

Hypothesis (from video review of task5): where the gripper lands on the bowl
(left vs right half) is set by the initial layout, and the policy places by
gripper position without compensating for the in-hand offset — so an
off-center grasp becomes an off-center placement and misses the 3cm success
predicate. This script turns that into numbers, from data already on disk
(per-step MuJoCo state dumps produced by attribution_probe.py --dump-states).

Modes:
  inits     Print the target-bowl / ramekin / plate positions for every fixed
            LIBERO init state of the task. Answers: is init3's arrangement an
            outlier among the 10 layouts?
  rollouts  For each states.npz under --states-glob: detect the grasp moment
            (target bowl lifted while gripper is close), report the xy offset
            of gripper vs bowl center at that moment, and the final bowl-to-
            plate-center distance (the quantity the predicate thresholds).

No policy and no rendering are needed; states are pushed straight into the
simulator (set_state + forward).

Examples:
  python scripts/grasp_offset_analysis.py --task-id 5 --mode inits
  python scripts/grasp_offset_analysis.py --task-id 5 --mode rollouts \
      --states-glob "$VLA_EVAL_OUTPUT_DIR/attr_f_task5_init3/rollout_*/states.npz" \
      --out "$VLA_EVAL_OUTPUT_DIR/attr_f_task5_init3/grasp_offset.json"
"""

import argparse
import glob
import json

import numpy as np

from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

GRASP_LIFT_M = 0.03   # bowl raised this much above its start height ...
GRASP_NEAR_M = 0.10   # ... while the grip site is within this distance = grasp


def build_sim(suite: str, task_id: int):
    envs = make_env(LiberoEnvConfig(task=suite, task_ids=[task_id]), n_envs=1)
    le = next(iter(envs.values()))[task_id].envs[0].unwrapped
    le.reset(seed=0)
    sim = getattr(le._env, "sim", None) or le._env.env.sim
    return le, sim


def body_xy_z(sim, body: str):
    p = sim.data.get_body_xpos(body)
    return np.array([p[0], p[1]]), float(p[2])


def find_bodies(sim):
    names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
    bowls = sorted(n for n in names if n and "bowl" in n and n.endswith("main"))
    plates = [n for n in names if n and "plate" in n and n.endswith("main")]
    rams = [n for n in names if n and "ramekin" in n and n.endswith("main")]
    assert bowls and plates and rams, f"unexpected body names: {names}"
    return bowls, plates[0], rams[0]


def target_bowl(sim, bowls, ramekin):
    # The instruction's bowl is the one sitting on the ramekin at t=0.
    rxy, _ = body_xy_z(sim, ramekin)
    return min(bowls, key=lambda b: np.linalg.norm(body_xy_z(sim, b)[0] - rxy))


def set_state(sim, flat):
    sim.set_state_from_flattened(np.asarray(flat, dtype=np.float64))
    sim.forward()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--mode", choices=["inits", "rollouts"], required=True)
    p.add_argument("--states-glob", help="[rollouts] glob of states.npz files")
    p.add_argument("--out", help="optional json output path")
    args = p.parse_args()

    le, sim = build_sim(args.suite, args.task_id)
    bowls, plate, ramekin = find_bodies(sim)
    print(f"bodies | bowls={bowls} plate={plate} ramekin={ramekin}")
    report = []

    if args.mode == "inits":
        for i, st in enumerate(le._init_states):
            set_state(sim, st)
            tb = target_bowl(sim, bowls, ramekin)
            row = {"init": i, "target_bowl": tb}
            for label, body in [("bowl", tb), ("ramekin", ramekin), ("plate", plate)]:
                xy, z = body_xy_z(sim, body)
                row[label] = [round(float(xy[0]), 4), round(float(xy[1]), 4), round(z, 4)]
            row["bowl_minus_plate_xy"] = [round(float(a - b), 4) for a, b in zip(row["bowl"][:2], row["plate"][:2])]
            report.append(row)
            print(row)

    else:
        for path in sorted(glob.glob(args.states_glob)):
            states = np.load(path)["states"]
            set_state(sim, states[0])
            tb = target_bowl(sim, bowls, ramekin)
            _, z0 = body_xy_z(sim, tb)
            grasp = None
            for t, st in enumerate(states):
                set_state(sim, st)
                bxy, bz = body_xy_z(sim, tb)
                eef = sim.data.get_site_xpos("gripper0_grip_site")
                if bz - z0 > GRASP_LIFT_M and np.linalg.norm(eef - np.array([*bxy, bz])) < GRASP_NEAR_M:
                    grasp = {
                        "grasp_step": t,
                        # bowl center relative to grip site, world xy (meters):
                        "bowl_minus_eef_xy": [round(float(bxy[0] - eef[0]), 4), round(float(bxy[1] - eef[1]), 4)],
                    }
                    break
            set_state(sim, states[-1])
            bxy, bz = body_xy_z(sim, tb)
            pxy, pz = body_xy_z(sim, plate)
            row = {
                "rollout": path.split("/")[-2],
                "grasp": grasp,
                "final_bowl_minus_plate_xy": [round(float(bxy[0] - pxy[0]), 4), round(float(bxy[1] - pxy[1]), 4)],
                "final_bowl_plate_dist": round(float(np.linalg.norm(bxy - pxy)), 4),
                "final_bowl_z_minus_plate_z": round(bz - pz, 4),
            }
            report.append(row)
            print(row)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
