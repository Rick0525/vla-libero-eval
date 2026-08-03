"""Second-carrier probe: warmstart-zeroing + RNG-pinning at every episode boundary (2026-08-03).

Round 2 of the cross-episode context case. Round 1 (oft_warmstart_probe.py)
closed e14 (carrier = boundary qacc_warmstart) but left e36 open: with the
warmstart zeroed in BOTH protocols, e36 still fails sequentially (x2) and
succeeds under the single-episode mirror (x2). The remaining known
deterministic difference between the two protocols at a boundary is the RNG
stream position (np/torch/random advance with each reset/rollout).

Intervention here: after reset + set_init_state + warmstart zeroing, re-seed
all RNG streams to the same fixed state (set_seed_everywhere(cfg.seed)) so the
rollout-time RNG stream is identical across episodes AND protocols.
Reading: if the sequential arm's e36 now matches the single arm, the second
carrier is the RNG stream; if the arms still disagree, the carrier sits below
the Python process state we can pin (renderer / driver / process history) and
the case closes with a boundary verdict.

Usage: OFT_CHECKPOINT=<dir> python oft_ws_rngpin_probe.py <task_id> <out_file> <ep> [<ep> ...]
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
        set_seed_everywhere(cfg.seed)
        fo.write(f"EP {ep} RNGPIN np={h(np.random.get_state()[1])}\n")
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
print("WSRNG_PROBE_DONE", flush=True)
