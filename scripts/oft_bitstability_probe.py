"""Bit-stability probe for the OpenVLA-OFT inference path.

Context: our 2026-08-02 determinism check re-ran the official OFT LIBERO-Spatial
eval with identical weights/config/invocation and observed 7/500 episode-outcome
flips (specialist arm, same GPU). This probe isolates the policy forward pass:
it feeds one fixed synthetic observation through the exact official assembly
(initialize_model + get_action) 10x in-process and hashes the action-chunk bytes.
Run the script twice (two processes) and compare PROCESS_HASH to also test
cross-process stability.

Observed result on our stack (A800, torch per official OFT env): the forward
pass is bit-stable both in-process and across processes -- the closed-loop
nondeterminism therefore lives in the env/rendering loop, not the policy head.

Usage (from the openvla-oft repo root, its venv active):
    OFT_CHECKPOINT=/path/to/openvla-oft-libero-spatial python oft_bitstability_probe.py
"""
import hashlib
import os

import numpy as np

from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.robot_utils import set_seed_everywhere, get_action, get_image_resize_size

cfg = GenerateConfig(
    pretrained_checkpoint=os.environ["OFT_CHECKPOINT"],
    task_suite_name="libero_spatial",
    center_crop=True,
    use_wandb=False,
)
set_seed_everywhere(cfg.seed)
model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
resize_size = get_image_resize_size(cfg)
if isinstance(resize_size, int):
    resize_size = (resize_size, resize_size)
print(f"resize_size={resize_size}", flush=True)

rng = np.random.default_rng(0)
base_obs = {
    "full_image": rng.integers(0, 256, (*resize_size, 3), dtype=np.uint8),
    "wrist_image": rng.integers(0, 256, (*resize_size, 3), dtype=np.uint8),
    "state": rng.standard_normal(8),
}
task_label = "pick up the black bowl on the ramekin and place it on the plate"

hashes = []
for i in range(10):
    obs = {k: v.copy() for k, v in base_obs.items()}
    actions = get_action(
        cfg, model, obs, task_label,
        processor=processor, action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        use_film=cfg.use_film,
    )
    arr = np.stack([np.asarray(a, dtype=np.float64) for a in actions])
    h = hashlib.md5(arr.tobytes()).hexdigest()
    hashes.append(h)
    print(f"call {i}: {h} first={arr.flatten()[:3]}", flush=True)

print("IN_PROCESS_BITSTABLE:", len(set(hashes)) == 1)
print("PROCESS_HASH:", hashes[-1])
