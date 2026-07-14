#!/usr/bin/env bash
# Pre-finetune baseline: smolvla_base (no LIBERO finetuning) on LIBERO-Spatial.
# Protocol: official LeRobot LIBERO docs — 10 episodes per task, batch_size=1,
# fixed seed. https://huggingface.co/docs/lerobot/main/en/libero
#
# Machine-specific paths come from env.local.sh (see env.example.sh).
# Overridable knobs: EVAL_N_EPISODES (default 10), EVAL_TASK_IDS (default all),
# EVAL_SEED (default 1000), RUN_TAG (suffix for the output dir).
set -euo pipefail

MODEL_PATH="${VLA_MODELS_DIR:?set VLA_MODELS_DIR (see env.example.sh)}/smolvla_base"
RUN_TAG="${RUN_TAG:-full}"
OUTPUT_DIR="${VLA_EVAL_OUTPUT_DIR:?set VLA_EVAL_OUTPUT_DIR}/smolvla_base_spatial_${RUN_TAG}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

EXTRA_ARGS=()
if [[ -n "${EVAL_TASK_IDS:-}" ]]; then
  EXTRA_ARGS+=("--env.task_ids=${EVAL_TASK_IDS}")
fi

lerobot-eval \
  --policy.path="${MODEL_PATH}" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes="${EVAL_N_EPISODES:-10}" \
  --env.max_parallel_tasks=1 \
  --seed="${EVAL_SEED:-1000}" \
  --output_dir="${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
