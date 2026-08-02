"""First-divergence probe for OFT closed-loop nondeterminism (2026-08-02).

Runs one task (a range of episodes sequentially on ONE env instance, mirroring
the official run_task loop incl. cross-episode solver state) and writes one
line of md5 digests per policy step:
    ep t sim=<mujoco full state> st=<8-dim policy state> img=<raw agentview>
         wri=<wrist img> act=<executed action>
Run the script twice (two processes); the first differing line, and which
field(s) differ on it, localizes where run-to-run divergence enters:
  act first (sim/st/img same)  -> policy side
  img/wri first (sim same)     -> rendering
  sim first (prior lines same) -> physics env.step

Usage: OFT_CHECKPOINT=<dir> python oft_divergence_probe.py <task_id> <ep_start> <ep_end> <out_file>
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

task_id, ep_start, ep_end, out_path = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

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

sim_handle = getattr(env, "sim", None)
if sim_handle is None:
    sim_handle = getattr(getattr(env, "env", None), "sim", None)

def h(x):
    return hashlib.md5(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()[:10]

with open(out_path, "w") as fo:
    for ep in range(ep_start, ep_end + 1):
        env.reset()
        obs = env.set_init_state(initial_states[ep])
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
print("PROBE_DONE", flush=True)
