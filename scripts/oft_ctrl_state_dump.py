"""Last-hop probe: does controller/robot Python-side state carry the RNG imprint
past set_init_state? (2026-08-03, model-free)

env.reset() consumes np.random for placements + robot init-qpos randn noise.
set_init_state overwrites all mjData qpos/qvel, but the robot/controller
PYTHON objects were initialized around the noised qpos and are not re-pinned.
Dump robot.init_qpos and all small float arrays in the controller object at
the e14 boundary under two histories (fresh-ish A1/A2 vs 13-episode context B),
plus mjData qpos as the known-pinned control. Fields differing across histories
while qpos is bit-equal are the carrier's last hop into the rollout.
"""
import hashlib

import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(7)
inits = suite.get_task_init_states(7)
env, _ = get_libero_env(task, "openvla", resolution=256)

def h(x):
    return hashlib.md5(np.ascontiguousarray(np.asarray(x, dtype=float)).tobytes()).hexdigest()[:10]

def get_sim():
    s = getattr(env, "sim", None)
    if s is None or not hasattr(s, "data"):
        s = getattr(env, "env", env).sim
    return s

def boundary_dump():
    env.reset()
    env.set_init_state(inits[14])
    inner = getattr(env, "env", env)
    r = inner.robots[0]
    out = {"mjdata.qpos": h(get_sim().data.qpos), "robot.init_qpos": h(r.init_qpos)}
    c = getattr(r, "controller", None) or getattr(r, "composite_controller", None)
    for k, v in sorted(vars(c).items()):
        if isinstance(v, np.ndarray) and v.dtype.kind == "f" and 0 < v.size <= 64:
            out[f"ctrl.{k}"] = h(v)
    return out

A1 = boundary_dump()
A2 = boundary_dump()
for ep in range(13):
    env.reset()
    env.set_init_state(inits[ep])
    for _ in range(60):
        env.step(get_libero_dummy_action("openvla"))
B = boundary_dump()

print(f"{'field':28s} {'A1':>10s} {'A2':>10s} {'B':>10s}  verdict")
for k in A1:
    a1, a2, b = A1[k], A2.get(k, "?"), B.get(k, "?")
    verdict = ("PINNED (bit-equal everywhere)" if a1 == a2 == b else
               "PER-RESET FRESH DRAW (varies every reset)" if a1 != a2 else
               "HISTORY CARRIER (stable within, differs across)")
    print(f"{k:28s} {a1:>10s} {a2:>10s} {b:>10s}  {verdict}")
print("CTRLDUMP_DONE")
