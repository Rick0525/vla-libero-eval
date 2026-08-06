#!/usr/bin/env bash
# Validation eval (Rick approved 8/5 afternoon): SmolVLA 30k, task5 only,
# 50 episodes, mujoco 3.8.1 + REGENERATED pre-settled init states
# (reseated_381_v2.npz — objects settled under 3.8.1, robot dofs pinned,
# zero velocities, time=0). Protocol otherwise a verbatim copy of the 7/30
# leaderboard run (which scored task5 = 14/50 on the official floating-bowl
# inits under this same mujoco 3.8.1).
# If SR recovers toward the 3.2.7 level (40/50), the task5 collapse is fully
# explained by the broken "bowl on the ramekin" premise, not by
# in-manipulation physics differences.
# The official init file is backed up and restored on exit, whatever happens.
set -uo pipefail

PY=$HOME/vla_lab/.venv/bin/python
export INIT=$HOME/vla_lab/.venv/lib/python3.12/site-packages/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.pruned_init
BAK=$INIT.orig_20260805

restore() {
  if [ -f "$BAK" ]; then
    mv -f "$BAK" "$INIT"
    echo "[restore] official task5 init file restored"
  fi
}
trap restore EXIT

source "$HOME/vla_lab/scripts/env.sh"
source /mnt/hdd16t/rick/vla_lab/vla-eval-harness/scripts/env.local.sh
export CUDA_VISIBLE_DEVICES=2
export MUJOCO_GL=egl

MJV=$("$PY" -c 'import mujoco; print(mujoco.__version__)')
echo "[preflight] mujoco=$MJV gpu=$CUDA_VISIBLE_DEVICES"
[ "$MJV" = "3.8.1" ] || { echo "PREFLIGHT_FAIL mujoco=$MJV"; exit 1; }

cp -n "$INIT" "$BAK"
"$PY" - <<'EOF'
import os
import numpy as np
import torch

src = np.load("/mnt/hdd16t/rick/vla_lab/eval_runs/mj_settle_probe_20260805/reseated_381_v2.npz")["states"]
p = os.environ["INIT"]
orig = torch.load(p, weights_only=False)
assert src.shape == orig.shape and src.dtype == orig.dtype, (src.shape, orig.shape)
torch.save(src, p)
print("[swap] task5 init file -> pre-settled states", src.shape)
EOF

echo "[eval] SmolVLA 30k task5x50 @mj381 + reseated inits $(date +%F_%T)"
cd /mnt/hdd16t/rick/vla_lab/vla-eval-harness
TRAIN_RUN=smolvla_spatial_b64_30k CKPT_STEP=030000 EVAL_N_EPISODES=50 \
  EVAL_TASK_IDS='[5]' EVAL_TAG=reseat381_20260805 bash scripts/eval_checkpoint_spatial.sh
rc=$?
echo "EVAL_DONE rc=$rc $(date +%F_%T)"
