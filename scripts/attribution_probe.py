#!/usr/bin/env python
"""Attribution probe for three-layer failure attribution experiments.

Two modes (see results/attribution_framework_zh.md for the experiment design):

  rollout  N closed-loop rollouts from ONE pinned LIBERO init state, with a
           fresh policy-noise seed per rollout. Optionally dumps the flattened
           MuJoCo sim state at every control step. Covers Exp F (fluke test:
           same layout, resampled noise) and oracle-state capture for Exp T.

  inject   N rollouts that start from a dumped mid-trajectory sim state
           ("takeover"): the policy inherits a state produced by another
           rollout (e.g. just after a successful grasp) and must finish the
           task. Covers Exp T. MuJoCo's get_state() does NOT include actuator
           ctrl, and a rim grip needs the servo to keep squeezing: after a
           reset the gripper ctrl creeps open and pries the restored fingers
           off the held object within ~2-4 steps (measured, 13/13 drops;
           neither close commands nor pinning ctrl at the finger position —
           zero squeeze force — prevents it). So rollout mode dumps the full
           ctrl vector alongside each state, and inject mode restores it
           verbatim — the only exact reconstruction of a mid-grasp actuator
           state. Optional --settle-steps then hold with a closing dummy.

Protocol parity with lerobot-eval: identical make_env / make_policy /
processor stack and the same rollout() loop; n_action_steps defaults to 1
(official protocol); TF32 and cudnn.benchmark match eval_main.

Examples (inside the project venv, after sourcing env.sh + env.local.sh):

  # Exp F: task5 init-state 3, 10 rollouts, policy seeds 2000..2009
  python scripts/attribution_probe.py --mode rollout --task-id 5 \
      --init-index 3 --n-rollouts 10 --base-seed 2000 \
      --out-dir "$VLA_EVAL_OUTPUT_DIR/attr_f_task5_init3"

  # Oracle capture: task5 init-state 6, one rollout, dump per-step states
  python scripts/attribution_probe.py --mode rollout --task-id 5 \
      --init-index 6 --n-rollouts 1 --base-seed 3000 --dump-states \
      --out-dir "$VLA_EVAL_OUTPUT_DIR/attr_t_capture_init6"

  # Exp T: take over from step 120 of that capture
  python scripts/attribution_probe.py --mode inject --task-id 5 \
      --state-file "$VLA_EVAL_OUTPUT_DIR/attr_t_capture_init6/rollout_000/states.npz" \
      --state-step 120 --n-rollouts 10 --base-seed 4000 \
      --out-dir "$VLA_EVAL_OUTPUT_DIR/attr_t_takeover_init6s120"
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed

REVIEW_FPS = 30  # review-video framerate; purely cosmetic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["rollout", "inject"], required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--model-path", default=None,
                   help="pretrained_model dir; default derives from $VLA_TRAIN_OUTPUT_DIR + --train-run/--ckpt-step")
    p.add_argument("--train-run", default="smolvla_spatial_b64_30k")
    p.add_argument("--ckpt-step", default="030000")
    p.add_argument("--n-action-steps", type=int, default=1, help="official protocol = 1 (re-plan every step)")
    p.add_argument("--n-rollouts", type=int, default=10)
    p.add_argument("--base-seed", type=int, required=True,
                   help="rollout k uses seed base+k for BOTH policy sampling noise and the env RNG")
    p.add_argument("--init-index", type=int, help="[rollout mode] LIBERO init-state index to pin (= episode index)")
    p.add_argument("--dump-states", action="store_true",
                   help="[rollout mode] save the flattened MuJoCo state at every control step")
    p.add_argument("--state-file", help="[inject mode] states.npz produced by --dump-states")
    p.add_argument("--state-step", type=int, help="[inject mode] which dumped step to take over from")
    p.add_argument("--settle-steps", type=int, default=0,
                   help="[inject mode] closed-gripper settle steps after state restore (see docstring)")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "rollout" and args.init_index is None:
        raise SystemExit("--init-index is required in rollout mode")
    if args.mode == "inject" and (args.state_file is None or args.state_step is None):
        raise SystemExit("--state-file and --state-step are required in inject mode")

    model_path = args.model_path or os.path.join(
        os.environ["VLA_TRAIN_OUTPUT_DIR"], args.train_run, "checkpoints", args.ckpt_step, "pretrained_model"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Numerics context identical to lerobot_eval.eval_main.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    env_cfg = LiberoEnvConfig(task=args.suite, task_ids=[args.task_id])
    envs = make_env(env_cfg, n_envs=1)
    suite_envs = next(iter(envs.values()))
    venv = suite_envs[args.task_id]
    le = venv.envs[0].unwrapped  # the LiberoEnv instance

    policy_cfg = PreTrainedConfig.from_pretrained(model_path)
    policy_cfg.pretrained_path = model_path
    policy_cfg.n_action_steps = args.n_action_steps
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(policy_cfg.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    # Pin the sim start point. LiberoEnv advances init_state_id by
    # _reset_stride on every reset; stride 0 keeps it pinned across rollouts
    # (including the env's internal auto-reset on termination).
    if args.mode == "rollout":
        le.init_state_id = args.init_index
        le._reset_stride = 0
        start_desc = {"init_index": args.init_index}
    else:
        data = np.load(args.state_file)
        dumped = data["states"]
        if "ctrls" not in data:
            raise SystemExit("state file lacks 'ctrls' — re-capture the source with the current probe "
                             "(exact ctrl restore is required for takeover, see docstring)")
        ctrls = data["ctrls"]
        if not 0 <= args.state_step < len(dumped):
            raise SystemExit(f"--state-step {args.state_step} out of range [0, {len(dumped)})")
        le._init_states = dumped[args.state_step][None]
        le.init_state_id = 0
        le._reset_stride = 0
        le.num_steps_wait = 0  # settled manually below, after pinning the grip

        def _reset_holding(seed=None, _orig=le.reset, **kw):
            """reset, then restore the source run's full actuation state (see docstring).

            sim.data.ctrl alone is not enough: robosuite recomputes the gripper
            ctrl every step from the python-side `gripper.current_action` creep
            state (reset to open), overwriting any restored value on step 1 —
            measured as fingers springing 4.7mm -> 19mm before re-closing, too
            late for the bowl. Invert the dumped ctrl through the actuator
            range to recover current_action exactly, no sign-convention guess.
            """
            obs, info = _orig(seed=seed, **kw)
            sim = getattr(le._env, "sim", None) or le._env.env.sim
            sim.data.ctrl[:] = ctrls[args.state_step]
            gripper = le._env.robots[0].gripper
            ids = [sim.model.actuator_name2id(a) for a in gripper.actuators]
            rng = sim.model.actuator_ctrlrange[ids]
            bias, weight = 0.5 * (rng[:, 1] + rng[:, 0]), 0.5 * (rng[:, 1] - rng[:, 0])
            gripper.current_action = np.clip((ctrls[args.state_step][ids] - bias) / weight, -1.0, 1.0)
            raw_obs = None
            for _ in range(args.settle_steps):
                raw_obs, _reward, _done, _info = le._env.step([0.0] * 6 + [1.0])
            if raw_obs is not None:
                obs = le._format_raw_obs(raw_obs)
            return obs, info

        le.reset = _reset_holding
        start_desc = {"state_file": str(args.state_file), "state_step": args.state_step,
                      "settle_steps": args.settle_steps}

    # Per-step sim-state + ctrl tap, installed around the env's own step().
    state_sink: list[np.ndarray] = []
    ctrl_sink: list[np.ndarray] = []
    if args.dump_states:
        orig_step = le.step

        def step_with_dump(action):
            out = orig_step(action)
            # On termination LiberoEnv.step() auto-resets, so the sim already
            # holds a fresh init state — skip it (the terminal sim state is
            # unrecoverable from outside; states[t] = post-step-t otherwise).
            if not out[2]:
                sim = getattr(le._env, "sim", None) or le._env.env.sim
                state_sink.append(np.asarray(sim.get_state().flatten(), dtype=np.float64))
                ctrl_sink.append(np.asarray(sim.data.ctrl, dtype=np.float64).copy())
            return out

        le.step = step_with_dump

    results = []
    for k in range(args.n_rollouts):
        seed = args.base_seed + k
        rdir = out_dir / f"rollout_{k:03d}"
        rdir.mkdir(exist_ok=True)
        state_sink.clear()
        ctrl_sink.clear()
        frames: list[np.ndarray] = []

        set_seed(seed)  # policy sampling noise (torch CPU+CUDA, numpy, python)
        with torch.no_grad():
            ret = rollout(
                venv, policy,
                env_preprocessor, env_postprocessor, preprocessor, postprocessor,
                seeds=[seed],
                render_callback=lambda _e: frames.append(le.render()),
            )

        success = bool(ret["success"].any().item())
        n_steps = int(ret["action"].shape[1])
        write_video(str(rdir / "review.mp4"), np.stack(frames), fps=REVIEW_FPS)
        if args.dump_states:
            np.savez_compressed(rdir / "states.npz", states=np.stack(state_sink), ctrls=np.stack(ctrl_sink))
        results.append({"rollout": k, "seed": seed, "success": success, "n_steps": n_steps})
        print(f"[{k + 1}/{args.n_rollouts}] seed={seed} success={success} steps={n_steps}", flush=True)

    summary = {
        "mode": args.mode,
        "suite": args.suite,
        "task_id": args.task_id,
        "task": le.task_description,
        "model_path": model_path,
        "n_action_steps": args.n_action_steps,
        "start": start_desc,
        "results": results,
        "n_success": sum(r["success"] for r in results),
        "n_rollouts": args.n_rollouts,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"== {summary['n_success']}/{args.n_rollouts} succeeded | {out_dir}/summary.json", flush=True)

    venv.close()


if __name__ == "__main__":
    main()
