"""Warmstart-zeroing intervention probe for the cross-episode context effect (2026-08-03).

Closes STATUS 8/02 unexpected finding #2 (single-episode protocol flips the
sequentially-locked t7 failures e14/e36) with a 2x2 design:
  natural x {sequential, single}: known — sequential fails {14,36,49} in 4/4
    runs, single-episode protocol succeeds on e14/e36 in 2/2 runs.
  zeroed  x {sequential, single}: this script — at every episode boundary
    (after reset + set_init_state) mjData.qacc_warmstart is zeroed, making the
    solver warmstart history-independent (MuJoCo cold-start semantics).
Reading: if zeroing collapses the sequential-vs-single outcome difference, the
carrier is the solver warmstart; if the zeroed arms still disagree, the carrier
lives elsewhere (renderer / process state) and warmstart is refuted.

Precheck facts this rests on (2026-08-03, model-free): env.reset() is a hard
reset (MjSim rebuilt), set_init_state pins qpos/qvel/ctrl/sim_state bit-exactly
across histories, yet boundary qacc_warmstart (= qacc, recomputed with the
previous warmstart as solver seed) differs per history — the only physics-side
field that does. Boundary dumps (pre-zero hash + max|.|) are logged per episode.
Per-step hash lines keep the oft_divergence_probe.py format for oft_div_analyze.py.

Usage: OFT_CHECKPOINT=<dir> python oft_warmstart_probe.py <task_id> <out_file> <ep> [<ep> ...]
"""
import hashlib
import os
import sys
from collections import deque

import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.run_libero_eval import (
    GenerateConfig, initialize_model, prepare_observation, process_action, TASK_MAX_STEPS,
)
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action
from experiments.robot.robot_utils import set_seed_everywhere, get_action, get_image_resize_size

task_id, out_path = int(sys.argv[1]), sys.argv[2]
episodes = [int(x) for x in sys.argv[3:]]

cfg = GenerateConfig(
    pretrained_checkpoint=os.environ["OFT_CHECKPOINT"],
    task_suite_name="libero_spatial",
    center_crop=True,
    use_wandb=False,
)
set_seed_everywhere(cfg.seed)
model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
resize_size = get_image_resize_size(cfg)

task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
task = task_suite.get_task(task_id)
initial_states = task_suite.get_task_init_states(task_id)
env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

def get_sim():
    # env.reset() hard-resets (rebuilds MjSim), so re-fetch every episode
    s = getattr(env, "sim", None)
    if s is None or not hasattr(s, "data"):
        s = getattr(env, "env", env).sim
    return s

def h(x):
    return hashlib.md5(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()[:10]

with open(out_path, "w") as fo:
    for ep in episodes:
        env.reset()
        obs = env.set_init_state(initial_states[ep])
        ws = get_sim().data.qacc_warmstart
        fo.write(f"EP {ep} WSBOUNDARY pre={h(ws)} absmax={float(np.abs(ws).max()):.6e}\n")
        ws[:] = 0.0
        assert float(np.abs(np.asarray(get_sim().data.qacc_warmstart)).max()) == 0.0
        action_queue = deque(maxlen=cfg.num_open_loop_steps)
        t = 0
        success = False
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            simh = h(env.get_sim_state()) if hasattr(env, "get_sim_state") else "na"
            observation, img = prepare_observation(obs, resize_size)
            if len(action_queue) == 0:
                actions = get_action(
                    cfg, model, observation, task_description,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)
            action = process_action(action_queue.popleft(), cfg.model_family)
            sth = h(observation["state"])
            imgh = h(img)
            wrih = h(observation["wrist_image"])
            acth = h(action)
            fo.write(f"{ep} {t} sim={simh} st={sth} img={imgh} wri={wrih} act={acth}\n")
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
        fo.write(f"EP {ep} success={success}\n")
        fo.flush()
print("WS_PROBE_DONE", flush=True)
