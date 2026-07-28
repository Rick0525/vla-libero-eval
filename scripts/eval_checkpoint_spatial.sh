#!/usr/bin/env bash
# Re-evaluate a finetuned checkpoint on LIBERO-Spatial under the SmolVLA
# paper's simulation protocol: re-predict after every executed action
# (--policy.n_action_steps=1, paper §4.3). The in-training curve was measured
# executing the full 50-action chunk open-loop, which the paper's own ablation
# (Table 13) shows costs ~35pp on Spatial — so curve numbers underestimate
# the checkpoints.
# Protocol otherwise per official LeRobot LIBERO docs — 10 episodes per task,
# batch_size=1, fixed seed. https://huggingface.co/docs/lerobot/main/en/libero
#
# Machine-specific paths come from env.local.sh (see env.example.sh). Knobs:
# CKPT_STEP (default last), N_ACTION_STEPS (default 1), TRAIN_RUN,
# EVAL_N_EPISODES (default 10), EVAL_TASK_IDS (default all), EVAL_SEED,
# EVAL_BATCH_SIZE (parallel env copies; episode seeds are assigned by episode
# index so results stay comparable across batch sizes), EVAL_USE_ASYNC
# (true/false override of eval.use_async_envs — this lerobot defaults it to
# true, but batch_size=1 silently downgrades to SyncVectorEnv, so the knob
# only matters at batch>1), EVAL_TAG (extra output-dir suffix so reruns don't
# collide).
set -euo pipefail

TRAIN_RUN="${TRAIN_RUN:-smolvla_spatial_official_100k}"
CKPT_STEP="${CKPT_STEP:-last}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MODEL_PATH="${VLA_TRAIN_OUTPUT_DIR:?set VLA_TRAIN_OUTPUT_DIR}/${TRAIN_RUN}/checkpoints/${CKPT_STEP}/pretrained_model"
OUTPUT_DIR="${VLA_EVAL_OUTPUT_DIR:?set VLA_EVAL_OUTPUT_DIR}/${TRAIN_RUN}_ckpt${CKPT_STEP}_n${N_ACTION_STEPS}${EVAL_TAG:+_${EVAL_TAG}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

EXTRA_ARGS=()
if [[ -n "${EVAL_TASK_IDS:-}" ]]; then
  EXTRA_ARGS+=("--env.task_ids=${EVAL_TASK_IDS}")
fi
if [[ -n "${EVAL_USE_ASYNC:-}" ]]; then
  EXTRA_ARGS+=("--eval.use_async_envs=${EVAL_USE_ASYNC}")
fi

lerobot-eval \
  --policy.path="${MODEL_PATH}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size="${EVAL_BATCH_SIZE}" \
  --eval.n_episodes="${EVAL_N_EPISODES:-10}" \
  --env.max_parallel_tasks=1 \
  --seed="${EVAL_SEED:-1000}" \
  --output_dir="${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
