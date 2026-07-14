#!/usr/bin/env bash
# Train SmolVLA on LIBERO per the official LeRobot recipe: pretrained SmolVLM2
# backbone (--policy.load_vlm_weights) + action expert trained from scratch on
# the LIBERO dataset, with in-training env eval on libero_spatial producing
# the success-rate curve. https://huggingface.co/docs/lerobot/main/en/libero
#
# Why no LoRA here: lerobot requires PEFT to start from a fully pretrained
# policy, and smolvla_base's SO-100 IO spec (3 cams, 6-dim state/action) is
# dimensionally incompatible with LIBERO (2 cams, 8-dim state, 7-dim action) —
# see results/smolvla_base_zero_shot.md. rename_map fixes names, not shapes.
#
# Machine paths come from env.local.sh (see env.example.sh). Knobs:
# TRAIN_BATCH_SIZE, TRAIN_STEPS, ENV_EVAL_FREQ, SAVE_FREQ, EVAL_N_EPISODES,
# LOG_FREQ, TRAIN_SEED, RUN_TAG.
set -euo pipefail

RUN_TAG="${RUN_TAG:-official}"
OUTPUT_DIR="${VLA_TRAIN_OUTPUT_DIR:?set VLA_TRAIN_OUTPUT_DIR}/smolvla_spatial_${RUN_TAG}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --batch_size="${TRAIN_BATCH_SIZE:-4}" \
  --steps="${TRAIN_STEPS:-100000}" \
  --log_freq="${LOG_FREQ:-100}" \
  --save_freq="${SAVE_FREQ:-5000}" \
  --eval.batch_size=1 \
  --eval.n_episodes="${EVAL_N_EPISODES:-1}" \
  --env_eval_freq="${ENV_EVAL_FREQ:-1000}" \
  --seed="${TRAIN_SEED:-1000}" \
  --output_dir="${OUTPUT_DIR}"
