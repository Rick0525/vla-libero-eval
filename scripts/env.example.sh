# Machine-specific configuration — copy to env.local.sh (gitignored) and fill in.
# Scripts in this repo read paths from the environment only; nothing is hardcoded.

export VLA_MODELS_DIR=/path/to/models          # contains smolvla_base/, openvla-oft-*/, ...
export VLA_EVAL_OUTPUT_DIR=/path/to/eval_runs  # evaluation outputs (json, videos)
export HF_LEROBOT_HOME=/path/to/datasets       # LeRobot dataset root (repo_id layout)
export MUJOCO_GL=egl                           # headless rendering on servers
