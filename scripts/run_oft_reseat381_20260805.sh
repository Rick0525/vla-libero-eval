#!/usr/bin/env bash
# OFT-family A/B (Rick approved 8/5 afternoon): GRPO checkpoint, greedy 500,
# mujoco 3.8.1 + REGENERATED pre-settled task5 init states.
# Comparators (8/4 four-run, same protocol): task5 = 6/50 @3.8.1 official
# inits, 49/50 @3.2.7 official inits. The other 9 tasks double as an internal
# control (should match the 8/4 mj381 arm within noise).
# The RLinf venv is pinned mujoco==3.3.0 (project convention) and its LIBERO
# lives at ~/vla_lab/LIBERO-oft. Both the mujoco version and the init file
# are restored on exit, whatever happens.
set -uo pipefail

RLPY=$HOME/vla_lab/rlinf/.venv/bin/python
UV=$HOME/.local/bin/uv
export INIT=$HOME/vla_lab/LIBERO-oft/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.pruned_init
BAK=$INIT.orig_20260805
MODEL=/mnt/hdd16t/rick/vla_lab/models/rlinf-openvlaoft-grpo-libero-spatial

restore() {
  if [ -f "$BAK" ]; then
    mv -f "$BAK" "$INIT"
    echo "[restore] official task5 init file restored"
  fi
  $UV pip install --python "$RLPY" --offline mujoco==3.3.0 \
    || $UV pip install --python "$RLPY" mujoco==3.3.0 \
    || { echo "RESTORE_FAILED mujoco=$($RLPY -c 'import mujoco;print(mujoco.__version__)')"; return; }
  echo "[restore] rlinf venv back to mujoco $($RLPY -c 'import mujoco;print(mujoco.__version__)')"
}
trap restore EXIT

$UV pip install --python "$RLPY" --offline mujoco==3.8.1 \
  || $UV pip install --python "$RLPY" mujoco==3.8.1
MJV=$($RLPY -c 'import mujoco; print(mujoco.__version__)')
echo "[preflight] rlinf venv mujoco=$MJV"
[ "$MJV" = "3.8.1" ] || { echo "PREFLIGHT_FAIL mujoco=$MJV"; exit 1; }

cp -n "$INIT" "$BAK"
"$RLPY" - <<'EOF'
import os
import numpy as np
import torch

src = np.load("/mnt/hdd16t/rick/vla_lab/eval_runs/mj_settle_probe_20260805/reseated_381_v2.npz")["states"]
p = os.environ["INIT"]
orig = np.asarray(torch.load(p, weights_only=False))
assert src.shape == orig.shape and src.dtype == orig.dtype, (src.shape, orig.shape)
torch.save(src, p)
print("[swap] task5 init file -> pre-settled states", src.shape)
EOF

echo "[eval] OFT-GRPO greedy 500 @mj381 + reseated task5 inits $(date +%F_%T)"
bash "$HOME/vla_lab/scripts/rlinf_libero_eval.sh" "$MODEL" oft-reseat381-mj381 100 False 5
rc=$?
echo "EVAL_DONE rc=$rc $(date +%F_%T)"
