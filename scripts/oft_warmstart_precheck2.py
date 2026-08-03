"""Differential boundary dump: which channel carries cross-episode context? (2026-08-03)

Reach episode-14's start boundary (reset + set_init_state(init[14])) under two
histories inside ONE process:
  A1/A2: fresh-ish context (back-to-back, gives the same-context noise floor)
  B:     after a 14-episode context (reset + set_init_state + 60 dummy steps each,
         a model-free stand-in for the sequential protocol's eps 0..13)
Bit-compare mjData fields, the flattened sim state, the numpy RNG position, and
the two returned camera images. Any field where |A1-B| exceeds the |A1-A2| floor
is a live carrier candidate; bit-equality across histories refutes that channel.

Usage: python oft_warmstart_precheck2.py
"""
import hashlib
import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action

task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = task_suite.get_task(7)
initial_states = task_suite.get_task_init_states(7)
env, task_description = get_libero_env(task, "openvla", resolution=256)

def get_sim():
    s = getattr(env, "sim", None)
    if s is None or not hasattr(s, "data"):
        s = getattr(env, "env", env).sim
    return s

def h(x):
    return hashlib.md5(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()[:10]

FIELDS = ["qacc_warmstart", "ctrl", "act", "qacc", "qfrc_applied", "sensordata", "qpos", "qvel"]

def boundary_dump(tag):
    env.reset()
    obs = env.set_init_state(initial_states[14])
    d = get_sim().data
    out = {"tag": tag}
    for f in FIELDS:
        v = getattr(d, f, None)
        out[f] = h(v) if v is not None and np.asarray(v).size else "absent"
    out["sim_state"] = h(env.get_sim_state()) if hasattr(env, "get_sim_state") else "na"
    out["time"] = float(d.time)
    out["rng"] = h(np.random.get_state()[1])
    out["img_agent"] = np.asarray(obs["agentview_image"]).copy()
    out["img_wrist"] = np.asarray(obs["robot0_eye_in_hand_image"]).copy()
    return out

def img_diff(a, b):
    da = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return int((da > 0).sum()), int(da.max()), a.size

A1 = boundary_dump("A1")
A2 = boundary_dump("A2")

for ep in range(14):
    env.reset()
    env.set_init_state(initial_states[ep])
    for _ in range(60):
        env.step(get_libero_dummy_action("openvla"))

B = boundary_dump("B")

print(f"{'field':16s} {'A1':>10s} {'A2':>10s} {'B':>10s}  verdict")
for f in FIELDS + ["sim_state", "time", "rng"]:
    a1, a2, b = A1[f], A2[f], B[f]
    same_floor = a1 == a2
    same_hist = a1 == b
    verdict = "DEAD (bit-equal across histories)" if same_hist else (
        "LIVE CARRIER (differs only across histories)" if same_floor else
        "noisy even within context (jitter, not a systematic carrier)")
    print(f"{f:16s} {str(a1):>10s} {str(a2):>10s} {str(b):>10s}  {verdict}")

for name in ["img_agent", "img_wrist"]:
    n12, m12, tot = img_diff(A1[name], A2[name])
    n1b, m1b, _ = img_diff(A1[name], B[name])
    print(f"{name}: floor A1-A2 {n12}/{tot} px (max {m12}); signal A1-B {n1b}/{tot} px (max {m1b})")

print("PRECHECK2_DONE")
