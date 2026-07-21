#!/usr/bin/env bash
# Train SmolVLA on LIBERO-Spatial with the paper-aligned recipe (§4.3 of the
# SmolVLA paper): global batch 64, bf16 mixed precision, torch.compile.
# Differences from train_smolvla_spatial.sh (the first 100k run):
#   - global batch 64 (paper) instead of 4 (LeRobot docs example)
#   - 30k steps: the SmolVLA optimizer preset decays lr over 30k steps, so a
#     100k run spends 70% of its budget at the 2.5e-6 floor; 30k ends exactly
#     when the schedule does. Paper: "can be trained for a much smaller number
#     of steps without sacrificing significant performance levels."
#   - bf16 autocast via `accelerate launch --mixed_precision=bf16` (the config
#     has no dtype field, so the launcher flag is the only entry point)
#   - NO in-training env eval: it cost 34% of wall-clock in the first run and
#     measured the wrong protocol anyway (n_action_steps=50). The eval daemon
#     (eval_daemon_spatial.sh) scores checkpoints on a separate GPU instead.
#   - DDP across NUM_GPUS processes; per-process batch = TRAIN_BATCH_SIZE_PER_GPU,
#     global batch = NUM_GPUS x per-process (verify smpl counter on first probe).
#
# Machine paths come from env.local.sh. Knobs: NUM_GPUS, TRAIN_BATCH_SIZE_PER_GPU,
# TRAIN_STEPS, SAVE_FREQ, NUM_WORKERS, LOG_FREQ, TRAIN_SEED, RUN_TAG, COMPILE.
set -euo pipefail

RUN_TAG="${RUN_TAG:-b64_30k}"
NUM_GPUS="${NUM_GPUS:-2}"
OUTPUT_DIR="${VLA_TRAIN_OUTPUT_DIR:?set VLA_TRAIN_OUTPUT_DIR}/smolvla_spatial_${RUN_TAG}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# 2026-07-21 re-test on CUDA-13 torch (2.11.0+cu130) after the IOMMU fix:
#   - dynamo ON (SmolVLA's inner hard-wired torch.compile): 600-step DDP probe
#     clean at 3.49 step/s (updt_s 0.279 / data_s 0.009) -> re-enabled.
#   - outer compile, 600-step DDP probe matrix (2026-07-22):
#       max-autotune:    dies in Triton autotune on sm_80 ("triton_mm Required:
#                        196608 Hardware limit: 166912" -> illegal memory access).
#                        Known inductor bug family (pytorch#95335), not the driver.
#       default:         clean, 4.33 step/s (updt_s 0.224)  <- new default
#       reduce-overhead: clean, updt_s 0.234 (cudagraphs likely skipped under DDP)
#       eager:           3.49 step/s (updt_s 0.279)
#     Compiled kernels change numerics slightly vs the eager W1 b64 run; success
#     rates are unaffected, bit-level loss comparisons across eras are not valid.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-0}"

MULTI_GPU_ARGS=()
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  MULTI_GPU_ARGS+=("--multi_gpu")
fi

accelerate launch \
  --num_processes="${NUM_GPUS}" \
  "${MULTI_GPU_ARGS[@]}" \
  --mixed_precision=bf16 \
  "$(command -v lerobot-train)" \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.compile_model="${COMPILE:-true}" \
  --policy.compile_mode="${COMPILE_MODE:-default}" \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --batch_size="${TRAIN_BATCH_SIZE_PER_GPU:-32}" \
  --steps="${TRAIN_STEPS:-30000}" \
  --num_workers="${NUM_WORKERS:-32}" \
  --prefetch_factor="${PREFETCH_FACTOR:-8}" \
  --log_freq="${LOG_FREQ:-100}" \
  --save_freq="${SAVE_FREQ:-2500}" \
  --seed="${TRAIN_SEED:-1000}" \
  --output_dir="${OUTPUT_DIR}"
