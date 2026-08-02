"""Instrumented single-episode OFT rollout with per-step action logging.

Purpose: test whether transient gripper releases seen in failure videos are
command-level events, and whether gripper-command reversals cluster at the
chunk-8 open-loop boundaries (phase 0 = first action of a freshly queried chunk).

Mirrors run_libero_eval.run_episode exactly, adding per-step logs:
  t, executed 7-dim action (post process_action; action[6]: -1=open, +1=close),
  chunk phase (0..7), gripper qpos (2-dim, from obs before the step).
Saves one .npz per episode + the replay video via save_rollout_video.

Findings on our stack (2026-08-02, task 7 both arms): successful episodes show
zero commanded open-flicks (transient loosening is physical slip under a CLOSE
command); failing episodes show open-commands clustering at chunk boundaries
(7/12 at phase 0; close-at-end-of-chunk -> open-at-start-of-next reversal pairs)
in repeated-regrasp segments. Also: episodes run under this single-episode
protocol can flip outcome vs the full-suite protocol (cross-episode env-state
carryover) -- single-episode probes are not substitutes for the full protocol.

Usage (from the openvla-oft repo root, its venv active):
    OFT_CHECKPOINT=<ckpt_dir> python oft_action_trace.py <task_id> <ep> [<ep> ...]
Optional: OFT_TRACE_OUT=<dir> (default ./oft_action_traces)
"""
import os
import sys
from collections import deque

import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.run_libero_eval import (
    GenerateConfig, initialize_model, prepare_observation, process_action, TASK_MAX_STEPS,
)
from experiments.robot.libero.libero_utils import (
    get_libero_env, get_libero_dummy_action, save_rollout_video,
)
from experiments.robot.robot_utils import set_seed_everywhere, get_action, get_image_resize_size

task_id = int(sys.argv[1])
episodes = [int(x) for x in sys.argv[2:]]
out_dir = os.environ.get("OFT_TRACE_OUT", "./oft_action_traces")
os.makedirs(out_dir, exist_ok=True)
ckpt = os.environ["OFT_CHECKPOINT"]
arm = os.path.basename(ckpt.rstrip("/"))

cfg = GenerateConfig(
    pretrained_checkpoint=ckpt,
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

for ep in episodes:
    env.reset()
    obs = env.set_init_state(initial_states[ep])
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    phase = -1
    replay_images = []
    rows = []  # t, phase, action(7), gripper_qpos(2)
    success = False
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue
        observation, img = prepare_observation(obs, resize_size)
        replay_images.append(img)
        if len(action_queue) == 0:
            actions = get_action(
                cfg, model, observation, task_description,
                processor=processor, action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )
            action_queue.extend(actions)
            phase = -1
        phase += 1
        action = action_queue.popleft()
        action = process_action(action, cfg.model_family)
        gq = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
        rows.append(np.concatenate(([t, phase], np.asarray(action, dtype=np.float64), gq)))
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1
    tag = f"{arm}_t{task_id}e{ep}"
    np.savez(os.path.join(out_dir, f"trace_{tag}.npz"),
             rows=np.stack(rows), success=success,
             columns="t,phase,dx,dy,dz,rx,ry,rz,grip,gq0,gq1")
    save_rollout_video(replay_images, 9000 + ep, success=success,
                       task_description=f"TRACE {tag} " + task_description, log_file=None)
    print(f"TRACE_DONE {tag} success={success} steps={len(rows)}", flush=True)
print("ALL_TRACES_DONE")
