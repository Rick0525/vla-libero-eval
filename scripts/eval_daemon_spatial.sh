#!/usr/bin/env bash
# Eval daemon: scores training checkpoints on a dedicated GPU as they appear,
# decoupling evaluation from training (the first 100k run lost 34% of its
# wall-clock to inline eval — and measured the wrong protocol anyway).
# Curve protocol: n_action_steps=10 (paper Table 13: n=10 matches n=1 within
# noise), 10 episodes per task, 10 parallel env copies for speed.
# First action is a one-off smoke check: task 0 of the first checkpoint at
# batch 1 vs batch 10 (same seeds) so parallel-env results can be compared.
# Waits while BLOCK_TMUX still owns the GPU; exits when the training session
# is gone and every checkpoint is scored. A failed eval retries after 10 min.
# Knobs: TRAIN_RUN, TRAIN_TMUX, BLOCK_TMUX, EVAL_GPU (default 2).
set -uo pipefail  # no -e: one failed eval must not kill the daemon

TRAIN_RUN="${TRAIN_RUN:-smolvla_spatial_b64_30k}"
TRAIN_TMUX="${TRAIN_TMUX:-smolvla_train_b64}"
BLOCK_TMUX="${BLOCK_TMUX:-eval_n1}"
EVAL_GPU="${EVAL_GPU:-2}"
CKROOT="${VLA_TRAIN_OUTPUT_DIR:?set VLA_TRAIN_OUTPUT_DIR}/${TRAIN_RUN}/checkpoints"
MARKS="${VLA_EVAL_OUTPUT_DIR:?set VLA_EVAL_OUTPUT_DIR}/${TRAIN_RUN}_daemon_marks"
mkdir -p "${MARKS}"

ready() {
  [[ -f "${CKROOT}/$1/pretrained_model/config.json" \
  && -f "${CKROOT}/$1/pretrained_model/model.safetensors" ]]
}

while true; do
  if tmux has-session -t "${BLOCK_TMUX}" 2>/dev/null; then sleep 300; continue; fi

  # Curve grid: 5000-step checkpoints only, 5 episodes/task. Smoke showed
  # eval.batch_size gives NO wall-clock speedup on LIBERO (envs step serially:
  # 449s vs 485s for 10 eps), so a full 10-ep/task point costs ~75 min and the
  # full 2500-grid would take ~16h. Finalists get a proper n=1 x 100-ep eval
  # separately once training frees a GPU.
  next=""
  for s in $(ls "${CKROOT}" 2>/dev/null | grep -E '^[0-9]+$' | sort); do
    [[ $((10#$s % 5000)) -eq 0 ]] || continue
    if [[ ! -f "${MARKS}/${s}.done" ]] && ready "${s}"; then next="${s}"; break; fi
  done

  if [[ -z "${next}" ]]; then
    if ! tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null; then
      echo "daemon: training gone and no pending checkpoints — exiting $(date)"
      break
    fi
    sleep 120; continue
  fi

  if [[ ! -f "${MARKS}/smoke.done" ]]; then
    echo "=== smoke: ckpt ${next} task0, batch 1 vs 10 — $(date) ==="
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" TRAIN_RUN="${TRAIN_RUN}" CKPT_STEP="${next}" \
      N_ACTION_STEPS=10 EVAL_BATCH_SIZE=1 EVAL_TASK_IDS='[0]' EVAL_TAG=smoke_b1 \
      bash scripts/eval_checkpoint_spatial.sh
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" TRAIN_RUN="${TRAIN_RUN}" CKPT_STEP="${next}" \
      N_ACTION_STEPS=10 EVAL_BATCH_SIZE=10 EVAL_TASK_IDS='[0]' EVAL_TAG=smoke_b10 \
      bash scripts/eval_checkpoint_spatial.sh
    touch "${MARKS}/smoke.done"
  fi

  echo "=== daemon: ckpt ${next} n=10 b=10 eps=5 start — $(date) ==="
  if CUDA_VISIBLE_DEVICES="${EVAL_GPU}" TRAIN_RUN="${TRAIN_RUN}" CKPT_STEP="${next}" \
       N_ACTION_STEPS=10 EVAL_BATCH_SIZE=10 EVAL_N_EPISODES=5 \
       bash scripts/eval_checkpoint_spatial.sh; then
    touch "${MARKS}/${next}.done"
    echo "=== daemon: ckpt ${next} done — $(date) ==="
  else
    echo "=== daemon: ckpt ${next} FAILED, retry in 10 min — $(date) ==="
    sleep 600
  fi
done
