# LIBERO-Spatial VLA Leaderboard (v1)

*2026-07-31 · 500 episodes per model · one shared, benchmark-pinned layout set · failure attribution included*

Three open-source VLA models (four checkpoints) evaluated on LIBERO-Spatial under a single
episode protocol, with per-model **official** inference configs disclosed column-by-column.
Beyond the headline scalar: per-task grids, hard-layout probes across the capacity axis, a
reproduction-gap accounting for π0.5, and video-verified failure mechanisms.

## Headline

| Model | Params | Checkpoint | Inference config (official) | Eval stack | Success (n=500) | 95% CI | Published ref |
|---|---|---|---|---|---|---|---|
| OpenVLA-OFT (specialist) | 7.5B | `moojink/openvla-7b-oft-finetuned-libero-spatial` | action chunk 8, open-loop | official OFT repo | **97.4** (487/500) | [95.6, 98.5] | 97.6 (paper) |
| OpenVLA-OFT (combined) | 7.5B | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` | action chunk 8, open-loop | official OFT repo | **97.4** (487/500) | [95.6, 98.5] | — |
| π0.5 | ~3.3B | `lerobot/pi05_libero_finetuned` | chunk 50, execute 10 | lerobot 0.6.0 | **93.8** (469/500) | [91.3, 95.6] | 97.0 @ n=100 (lerobot docs) |
| SmolVLA (our finetune) | 0.45B | b64 / 30k steps ([recipe](smolvla_spatial_finetune_report.md)) | n_action_steps=1 (re-plan every step) | lerobot 0.6.0 | **83.8** (419/500) | [80.3, 86.8] | ~90 (paper) |

Two findings that don't fit in one scalar:

- **Multi-task tax = 0.** OFT specialist and combined score identically (487/500 each) and
  their per-task grids differ by ≤2 episodes everywhere — the 4-suite co-trained checkpoint
  pays no measurable price on Spatial. Their few task-5 failures don't even overlap
  (episodes 33/39 vs 22/42): weight differences flip outcomes only on marginal layouts.
- **Wall clock is dominated by success rate and inference frequency, not parameter count.**
  7.5B OFT finishes 500 episodes in ~53 min (chunk-8 open-loop → 1/8 the inference calls;
  97% successes exit early); 0.45B SmolVLA takes ~2.5 h (re-plans every step, more long
  failure episodes); π0.5 ~60 min.

## Protocol

**Shared across all models:**

- LIBERO-Spatial, all 10 tasks × 50 episodes = 500 episodes.
- **Layouts are benchmark-pinned, not sampled:** episode *i* of a task resets the sim to
  init state *i* from the LIBERO benchmark's official `init_states` file. Both eval stacks
  index the same files, so all four checkpoints face the exact same 500 initial layouts.
  Success denominators are a fixed layout set, not random repetitions.
- Success = LIBERO's task predicate (e.g. `On`: direct contact + centers within 3 cm).

**Per-model (official configs kept, differences disclosed):**

- Inference configs as in the headline table — these are each model's published settings,
  not tuned by us.
- Eval stack: SmolVLA and π0.5 run in lerobot 0.6.0 (sync vector env, batch 1, seed 1000);
  OFT runs in its official repo environment (`cudnn.deterministic=True`).
- Episode cap: 280 steps (lerobot stack) vs 220 + 10 settle steps (OFT official).
  The asymmetry favors the lerobot-stack models if anything; OFT leads despite the shorter
  budget.
- SmolVLA has no official LIBERO-Spatial checkpoint; its row is our paper-aligned finetune
  (global batch 64, 30k steps). All other rows are unmodified released checkpoints.

## Per-task grid (successes / 50)

Task instructions are all "pick up the black bowl … and place it on the plate"; the
spatial clause is what varies:

| # | Spatial clause | SmolVLA | π0.5 | OFT spec. | OFT comb. |
|---|---|---|---|---|---|
| 0 | between the plate and the ramekin | 45 | 50 | 50 | 50 |
| 1 | next to the ramekin | 48 | 48 | 50 | 49 |
| 2 | from table center | 46 | 50 | 50 | 49 |
| 3 | on the cookie box | 47 | 49 | 49 | 50 |
| 4 | in the top drawer of the wooden cabinet | 42 | 46 | 47 | 48 |
| 5 | on the ramekin | **14** | 43 | 48 | 48 |
| 6 | next to the cookie box | 47 | 50 | 49 | 49 |
| 7 | on the stove | 48 | **40** | 47 | **46** |
| 8 | next to the plate | 41 | 49 | 49 | 49 |
| 9 | on the wooden cabinet | 41 | 44 | 48 | 49 |

The hardest task is **model-dependent**: task 5 craters SmolVLA (28%), task 7 is π0.5's
worst (80%) while being one of SmolVLA's best (96%), and OFT never drops below 92%.
See failure attribution below — the reversal has a mechanism.

## Hard-layout probes on the capacity axis

**Task 5 ("on the ramekin"), all 50 layouts:**
SmolVLA **28%** → π0.5 **86%** → OFT **96%**.

**Init-state 3 of task 5** — the single hardest layout we know (SmolVLA's historical
0/13 across all runs):

| Model | Result | Metric type |
|---|---|---|
| SmolVLA (base finetune) | 0/13 | hit rate (stochastic policy) |
| SmolVLA + targeted upsampling | 4/10 | hit rate |
| π0.5 | **5/10** (probe, policy seeds 2000–2009) | hit rate |
| OFT specialist / combined | 1/1 each | binary coverage (deterministic policy) |

Metric-type note: SmolVLA and π0.5 sample actions (flow matching), so a pinned layout has
a hit *rate*; OFT's L1-regression head is deterministic, so the same layout gives one
binary answer — the two numbers are not directly comparable, which is itself a protocol
lesson for cross-model probes. The poison layout stays genuinely hard for π0.5 (5/10),
so the capacity/data ladder is monotone even at the hardest layout we know.

## π0.5 reproduction gap: a three-line accounting

Our 93.8 vs the widely-quoted 97.5 decomposes as:

1. **Aggregation:** 97.5 is a *four-suite average*; the Spatial-only reference is **97.0**,
   measured at **n=100** (10 episodes/task, official protocol).
2. **Statistics:** 469/500 vs 97/100 → two-proportion z = 1.26, p = 0.21 — not
   significant. The n=100 point estimate carries a Wilson 95% CI of [91.5, 99.0], which
   contains our entire measurement.
3. **Layout-subset bias:** the official n=100 protocol only ever visits init states 0–9.
   8 of π0.5's 10 task-7 failure layouts have index ≥ 10 — under the official protocol
   task 7 scores 8/10; over all 50 pinned layouts it is 40/50. The reference number
   structurally cannot see most of the hard layouts.

Community reproductions of this checkpoint family report far lower numbers
(e.g. Spatial 77% in [lerobot#2114](https://github.com/huggingface/lerobot/issues/2114));
93.8 sits at the high end of the reproduction distribution.

## Failure attribution: π0.5 (all 29 failures video-reviewed, two independent runs)

We re-ran the four bleeding tasks (4/5/7/9) with every episode recorded
(`scripts/lerobot_eval_fullvideo.py`) and reviewed all failures frame-by-frame:

| Task | Fails | Overlap across 2 runs | Mechanism (video verdict) |
|---|---|---|---|
| 7 — on the stove | 10 | **10/10 identical** | **Referent confusion**: picks the *cabinet-top* bowl (10/10), places it neatly on the plate, predicate never fires |
| 9 — on the cabinet | 6 | 5/6 | Mirror confusion: picks the *stove* bowl |
| 5 — on the ramekin | 9 | 4 | Grasp failure (8/9 never lift the bowl; the one lift grabs the bowl's left rim and drops it on the plate's edge) |
| 4 — in the top drawer | 4 | 1/4 | Grasp failure (0/4 ever lift the bowl) |

Two structurally different failure modes, and **failure reproducibility is the mechanism
fingerprint**: referent confusion is a discrete decision error — given the layout it
recurs deterministically (10/10 identical failure sets across independent runs of a
*stochastic* policy) — while grasp failures live at contact-geometry margins where
sampling noise flips outcomes (low overlap). One can classify the mechanism family
from failure-set overlap alone, before watching a single video.

**Cross-model picture — capacity changes the failure mechanism mix; it does not
monotonically fix everything:**

- 0.45B SmolVLA fails at **contact geometry** (task-5 grasp-side bias; causal chain
  established by intervention, Fisher p ≈ 0.0098 — see
  [attribution report](attribution_framework_zh.md)) but makes no stove/cabinet
  confusion on task 7 (48/50).
- ~3.3B π0.5 largely fixes contact geometry (43/50 on task 5) but introduces a
  **symmetric spatial-referent confusion** between two adjacent elevated surfaces
  ("on the stove" ↔ "on the wooden cabinet") that the smaller model does not make.
  Its residual task-5 failures are the same grasp-margin family as SmolVLA's
  (offset grip → rim placement).
- 7.5B OFT covers both (no task below 46/50); its few failures are unreviewed.

## Limitations & open questions

- Headline numbers are single runs (the 4-task π0.5 re-run replicated 171/200 vs
  173/200, and OFT is deterministic, but SmolVLA/π0.5 totals carry sampling noise on
  top of the CIs shown).
- SmolVLA's own task 9 (41/50) failures are not yet video-reviewed — "0.45B makes no
  stove/cabinet confusion" is verified on task 7 only.
- OFT's light task-7/9 failures are unreviewed; OFT-combined is evaluated on Spatial
  only so far.
- The two eval stacks differ beyond inference configs (episode caps 280 vs 220+10,
  env library versions); we kept each model's official stack by design and disclose
  rather than normalize.
- Suites Object / Goal / Long and GR00T are not yet included.

## Reproduce

- SmolVLA / π0.5: `lerobot-eval` (lerobot 0.6.0) with the configs above; all-episode
  video via `scripts/lerobot_eval_fullvideo.py` (`EVAL_RENDER_ALL_N` knob).
- Pinned-layout probes: `scripts/attribution_probe.py 
  --mode rollout --task-id 5 --init-index 3 --model-path <ckpt> --n-action-steps <official>`.
- OFT: official `openvla-oft` repo eval script, unmodified protocol.
