# SmolVLA Fine-tuning on LIBERO-Spatial — Report

**Headline: 82.0% success rate** on LIBERO-Spatial (official protocol: `n_action_steps=1`, 10 tasks × 10 episodes, seed 1000), trained in 4h02m on 2×A800. Paper reference for SmolVLA-0.45B: 90.

## Setup

- **Policy**: SmolVLA 0.45B — pretrained SmolVLM2-500M backbone (first 16 layers, frozen; `train_expert_only=true`), action expert trained from scratch, chunk size 50, 512×512 inputs.
- **Data**: `HuggingFaceVLA/libero`, libero_spatial subset — 1,693 trajectories.
- **Framework**: lerobot v0.6.0. Training: `scripts/train_smolvla_spatial_b64.sh`; evaluation: `scripts/eval_checkpoint_spatial.sh`.

## Two runs, one lesson

| | Run 1 — docs-example recipe | Run 2 — paper-aligned recipe |
|---|---|---|
| Global batch | 4 | 64 (32 × 2 GPUs, DDP) |
| Steps | 100k | 30k |
| Samples seen | 0.4M (~1.5 epochs) | 1.9M (~7 epochs) |
| Precision | fp32 | bf16 (`accelerate --mixed_precision=bf16`) |
| Wall clock | 8h53m (incl. inline eval) | 4h02m |
| Final loss | 0.434 | 0.237 |
| **Success (n=1 × 100 eps)** | **43.0%** (100k ckpt; 60k ckpt: 38.0%) | **82.0%** (30k ckpt) |

Run 1 followed the lerobot LIBERO docs' example training command verbatim. Three compounding issues, in order of discovery:

1. **Evaluation protocol**: the training-time curve was scored with `n_action_steps=50` (execute the whole action chunk open-loop). The paper's own ablation (Table 13) shows 54% at n=50 vs 89% at n=1 on Spatial — a ~35pp penalty from a purely evaluation-time parameter.
2. **Batch size**: 4 vs the paper's 64 (§4.3) — 16× fewer samples at equal step count.
3. **LR schedule**: lerobot's training preset anneals the cosine LR to its floor (2.5e-6) by step 30k regardless of `--steps`. In the 100k run, loss moved only 0.442 → 0.424 over the final 60k steps — the last 70% of the run was near-idle compute.

Run 2 fixes all three: batch 64, 30k steps (the scheduler's natural horizon; the paper itself notes step count can be reduced substantially with little loss), formal evaluation at n=1.

## Learning curve (Run 2)

Scored by an evaluation daemon on a separate GPU (`n_action_steps=10`, 50 episodes/point):

| step | 5k | 10k | 15k | 20k | 25k | 30k |
|---|---|---|---|---|---|---|
| success % | 52 | 48 | 72 | 68 | 74 | 74 |

The curve plateaus after ~15k steps, consistent with 7 epochs over 1,693 trajectories and the LR floor. Notably, the 5k checkpoint (0.32M samples) already beats Run 1's final 100k checkpoint (0.4M samples).

## Final evaluation (official protocol, n=1 × 100 episodes)

- ckpt 30k: **82.0%** — 95% CI [74.5, 89.5] (±7.5pp at n=100)
- ckpt 25k: 84.0% — a statistical tie (2 episodes apart)

We report **30k / 82.0%** as the headline. Reporting 25k's 84% would incur winner's-curse bias — using the same evaluation for both checkpoint selection and reporting inflates the winner's score; it would only be reportable after re-evaluation with fresh seeds.

**Gap to the paper's 90**: not statistically resolvable at n=100 (the CI's upper edge is 89.5). A verdict either way needs ≥400 episodes (±3.8pp) plus item-by-item protocol alignment. Open.

## Per-task results (ckpt 30k)

All 10 LIBERO-Spatial tasks share the same manipulation — "pick up the black bowl … and place it on the plate" — and differ only in the spatial expression identifying which of two identical black bowls is the target.

| Task | Spatial expression | Success |
|---|---|---|
| 0 | between the plate and the ramekin | 9/10 |
| 1 | next to the ramekin | 9/10 |
| 2 | from table center | 9/10 |
| 3 | on the cookie box | 10/10 |
| 4 | in the top drawer of the wooden cabinet | 9/10 |
| **5** | **on the ramekin** | **2/10** |
| 6 | next to the cookie box | 10/10 |
| 7 | on the stove | 10/10 |
| 8 | next to the plate | 6/10 |
| 9 | on the wooden cabinet | 8/10 |

## Failure analysis (open)

task5 is the outlier — 2/10 at 30k, and also the worst task at 25k (4/10): the weakness is systematic across checkpoints, not evaluation noise. Hypotheses were registered **before** watching any videos (pre-registration guards against hindsight bias; see `failure_videos/ANNOTATION_zh.md`):

- **H1 — grounding**: approaches the wrong (distractor) bowl; motion itself fluent.
- **H2 — grasp geometry**: correct referent, but a bowl stacked on a ramekin needs an atypical grasp (elevated, unstable base) — missed or knocked off.
- **H3 — placement**: successful grasp, failed placement on the plate.
- **H4 — timeout/dithering**: oscillates between the two bowls without committing.

Pre-registered prediction: **H2**. An 8/10 failure rate is too high for random two-way referent confusion (~50% expected), and "bowl stacked on another bowl" is a tail configuration in the training data. Verdict pending video annotation.

## Reproducibility notes

- Train and eval seed: 1000. Eval seeds are assigned per episode index, independent of `--eval.batch_size` — batch settings do not change the evaluated initial states.
- `--eval.batch_size` yields no speedup on LIBERO (environments step serially); `--env.max_parallel_tasks` must stay 1 (the implementation shares one stateful policy object across a thread pool).
- Two machine-specific workarounds (NCCL P2P disabled; TorchDynamo disabled globally — lerobot's SmolVLA hardcodes an internal `torch.compile` not governed by `--policy.compile_model`) are documented inline in `scripts/train_smolvla_spatial_b64.sh`.
