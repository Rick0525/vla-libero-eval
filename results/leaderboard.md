# LIBERO-Spatial VLA Leaderboard

*English · [中文](leaderboard_zh.md)*

*2026-07-31 first published · 2026-08-06 main table calibrated to common physics · 2026-08-21 rewritten for readers · 500 episodes per model · all four rows on MuJoCo 3.2.7 · failure attribution by video review (per-model coverage disclosed)*

Three open-source VLA models (four checkpoints) evaluated on LIBERO-Spatial under a unified protocol: 500 episodes per model, the same set of benchmark-pinned initial layouts, MuJoCo 3.2.7 throughout, each model running its own published inference config—disclosed column by column. Beyond the headline number we report: per-task success rates, reconciliation against published reference values, video-verified failure attribution (coverage disclosed per model), and a methodological finding—**LIBERO's per-task numbers are highly sensitive to MuJoCo physics version** (see dedicated section).

## Headline

| Model | Params | Checkpoint | Inference config (official) | Eval stack | Success (n=500) | 95% CI | Published ref |
|---|---|---|---|---|---|---|---|
| OpenVLA-OFT (specialist) | 7.5B | `moojink/openvla-7b-oft-finetuned-libero-spatial` | action chunk 8, open-loop | official OFT repo | **97.4** (487/500) | [95.6, 98.5] | 97.6 (paper) |
| OpenVLA-OFT (combined) | 7.5B | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` | action chunk 8, open-loop | official OFT repo | **97.4** (487/500) | [95.6, 98.5] | — |
| π0.5 | ~3.3B | `lerobot/pi05_libero_finetuned` | chunk 50, execute 10 | lerobot 0.6.0 | **94.4** (472/500) | [92.0, 96.1] | 97.0 @ n=100 (lerobot docs) |
| SmolVLA (our finetune) | 0.45B | b64 / 30k steps ([recipe](smolvla_spatial_finetune_report.md)) | n_action_steps=1 (re-plan every step) | lerobot 0.6.0 | **87.4** (437/500) | [84.2, 90.0] | ~90 (paper) |

Two findings that don't fit in a single number:

- **The combined checkpoint pays no measurable cost on Spatial, and the result is robust to re-runs.** Specialist and combined were each evaluated twice in full: specialist 487/490, combined 487/489. Within the same checkpoint, the two runs differ by ≤±1 task-level episode; specialist vs combined differ by ≤±2—co-training on four suites does not drag down the Spatial score. The only qualitative difference surfaced by video review is an isolated case: the combined checkpoint heads for the wrong bowl on one layout of task 5 in both runs, while the specialist selects correctly on the same layout (1/500, single layout, not generalized; details in "Failure mechanisms").
- **Wall-clock time is dominated by success rate and inference frequency, not parameter count.** The 7.5B OFT finishes 500 episodes in ~53 min (chunk-8 open-loop → 1/8 the inference calls; 97% of episodes are successes that exit early). The 0.45B SmolVLA takes ~90 min (re-plans every step → 8× more inference calls, and more failure episodes that run to the step limit). π0.5 lands at ~62 min. (Wall-clock measured on a shared server; treat as order-of-magnitude.)

## Protocol

**Shared across all models:**

- LIBERO-Spatial, all 10 tasks × 50 episodes = 500 episodes.
- **Layouts are benchmark-pinned, not sampled.** Episode *i* of a task resets the simulator to init state *i* from the official LIBERO `init_states` file. Both eval stacks (lerobot and OFT's official repo, see headline table) index the same files, so all four checkpoints face the exact same 500 initial layouts. The success-rate denominator is a fixed layout set, not random repetitions. (Note: `init_states` stores a *pre-state*—the black bowl suspended ~11 cm above the ramekin, with physics completing the landing during reset. The semantics of the initial layout depend on physics completion, which is the root cause of the version-sensitivity finding below.)
- Success = LIBERO's task predicate (e.g. `On`: direct contact + center xy distance < 3 cm).

**Per-model (each model's official config preserved; differences disclosed):**

- Inference configs as in the headline table—these are each model's published settings; we did not tune them.
- Eval stack: SmolVLA and π0.5 run on lerobot 0.6.0 (sync vector env, batch 1, seed 1000, MuJoCo 3.2.7). OFT runs on its official repo (MuJoCo 3.2.7—the official dependency chain does not pin MuJoCo; at install time it resolved to a version in the healthy range. `cudnn.deterministic=True` constrains only cuDNN convolutions and does not guarantee closed-loop bit-reproducibility; see "Limitations").
- Episode step budget: 280 policy steps (lerobot stack) vs 220 (OFT official). Both stacks run the same 10 no-op physics-settling steps after `set_init_state`: lerobot inside `env.reset()` (`num_steps_wait=10` in `lerobot/envs/libero.py`, not counted in the 280), OFT explicitly in its eval loop (booked as 220 + 10). The settling phase is symmetric; the only asymmetry is the 280-vs-220 budget. If anything, this favors the lerobot-stack models—OFT leads despite the shorter budget.
- SmolVLA has no official LIBERO checkpoint. This row is our finetune following the paper's recipe (all four suites, 40 tasks, 1,693 demos, global batch 64, 30k steps), evaluated on Spatial. All other rows are unmodified published checkpoints.

## Per-task breakdown (successes / 50)

All task instructions follow "pick up the black bowl … and place it on the plate"; only the spatial clause varies:

| # | Spatial clause | SmolVLA | π0.5 | OFT spec. | OFT comb. |
|---|---|---|---|---|---|
| 0 | between the plate and the ramekin | 40 | 50 | 50 | 50 |
| 1 | next to the ramekin | 47 | 48 | 50 | 49 |
| 2 | from table center | 47 | 49 | 50 | 49 |
| 3 | on the cookie box | 45 | 50 | 49 | 50 |
| 4 | in the top drawer of the wooden cabinet | 39 | 46 | 47 | 48 |
| 5 | on the ramekin | 40 | 45 | 48 | 48 |
| 6 | next to the cookie box | 47 | 50 | 49 | 49 |
| 7 | on the stove | 48 | 40 | 47 | 46 |
| 8 | next to the plate | 43 | 50 | 49 | 49 |
| 9 | on the wooden cabinet | 41 | 44 | 48 | 49 |

**The hardest task is model-dependent.** SmolVLA's weak spots are spread across t0/t4/t5/t9 (78–82%). Task 7 is π0.5's sole deep failure (80%), yet it is one of SmolVLA's best (96%). OFT never drops below 92%. This reversal has a mechanistic explanation—see "Failure mechanisms." The one broadly shared weak spot, task 5 ("on the ramekin"), improves monotonically along the capacity axis (80% → 90% → 96%), but the slope is gentle, not a cliff.

## Reconciliation with published numbers

- **OFT:** Our 97.4 vs the paper's 97.6 (specialist, Spatial only)—consistent; the CI fully contains the reference. No published Spatial-only reference exists for the combined checkpoint.
- **SmolVLA:** Our 87.4 vs the paper's ~90. Apples-to-apples check: the paper's ~90 comes from a single model trained on all four suites (40 tasks, 1,693 demos, 100k steps), reported per suite at n=100. Our training recipe aligns (same full dataset, 30k steps—the paper notes training steps can be substantially reduced with minimal performance loss); our evaluation is 500 fixed-layout episodes vs the paper's 100. The paper does not record MuJoCo version; by publication date it falls in the healthy physics range. The 2.6 pp gap gives z = 0.73, p = 0.47—not statistically significant.
- **π0.5:** Our 94.4 vs the widely cited 97.5. Three lines of accounting:
  1. **Aggregation mismatch.** 97.5 is a four-suite average. The Spatial-only reference is **97.0**, measured at **n=100** (10 episodes/task, official protocol).
  2. **Statistics.** 472/500 vs 97/100 → two-proportion z = 1.07, p = 0.28—not significant. The n=100 point estimate has a Wilson 95% CI of [91.5, 99.0], which fully contains our measurement.
  3. **Layout-subset bias.** The official n=100 protocol only visits init states 0–9. Of π0.5's 10 task-7 failures, 8 have layout index ≥ 10—under the official protocol, task 7 scores 8/10; over all 50 pinned layouts it is 40/50. The reference number **structurally cannot see** most of the hard layouts.

  The reference does not record a MuJoCo version, but this checkpoint was released 2025-10, before the 3.4.0 fix, so it was necessarily tested in the healthy physics range—same as our main table. Separately: community reproductions of this checkpoint family report substantially lower numbers (e.g. Spatial 77% in [lerobot#2114](https://github.com/huggingface/lerobot/issues/2114)); our 94.4 sits at the high end of the reproduction distribution.

## Failure mechanisms

All three models' failures were reviewed on per-segment video, with coverage varying by model: OFT—every failure across both arms and both runs; π0.5—all failures in the four most failure-concentrated tasks; SmolVLA—all task 9 failures plus the recorded task 5 failures. Details in each subsection.

### π0.5: two distinct failure modes

The main run only records the first 10 episodes per task—insufficient for systematic video review. We re-ran the four worst tasks (4/5/7/9), 50 episodes each, under the same protocol with full recording (`scripts/lerobot_eval_fullvideo.py`). Seeds are assigned by within-task episode index, so the *N*-th episode maps to the same layout in both runs. The action sampling noise, however, comes from a per-process random stream seeded at the run level; since the re-run starts from task 4 while the main run starts from task 0, the stream positions differ—**each layout gets two independent trials.**

The table below counts failures from the full-video re-run (29 total). "Overlap" = how many of these failure layouts also failed in the main run. Overlap matters because π0.5 re-samples actions on each inference step (flow matching) and the two runs draw from different noise streams—same layout, two independent rolls. If the same layout fails both times, the failure is layout-determined, not luck.

> Note: the video corpus comes from a run under 3.8.1 physics (broken-premise era; see version-sensitivity section), hence 29 failures vs the main table's 28 under 3.2.7. The mechanistic conclusions carry over: π0.5 is nearly immune to the version effect (task 5: 43→45; six tasks unchanged; t2/t3/t8 each ±1; total 469→472), and the t7/t9 referent confusion has nothing to do with the task 5 premise.

| Task | Failures | Cross-run overlap | Mechanism (video verdict) |
|---|---|---|---|
| 7 — on the stove | 10 | **10/10 identical** | **Referent confusion**: picks the *cabinet-top* bowl (10/10), places it neatly on the plate; predicate never fires |
| 9 — on the cabinet | 6 | 5/6 | Mirror confusion: picks the *stove* bowl |
| 5 — on the ramekin | 9 | 4/9 | Grasp failure (8/9 never lift the bowl; the one lift grabs the left rim, drops it on the plate edge) |
| 4 — in the top drawer | 4 | 1/4 | Grasp failure (0/4 ever lift the bowl) |

The two failure modes differ in nature, and the difference is directly legible in overlap:

- **Referent confusion is a discrete decision error.** The wrong target is selected; the choice is layout-determined, luck-independent. Same layout, always wrong—task 7 has 10/10 identical failure sets across two independent runs of a *stochastic* policy.
- **Grasp failure is a marginal contact event.** Success hinges on millimeters; re-drawing the action once can flip the outcome. Hence low overlap—task 4 shares only 1/4.

Overlap therefore serves as a cheap screening signal: high overlap → the layout is **near-certain to fail** for this policy; low overlap → the failure is marginal luck. But the signal has a hard limit: **it cannot tell you *why* the layout is near-certain to fail**, because the cause could equally be an impossible grasp geometry. Mechanism identification ultimately requires watching the video.

On these 29 π0.5 failures, the two signals happen to align perfectly: all locked failures are referent confusion, all fluctuating failures are grasp failures. This alignment is a finding verified by video, not a theorem derived from overlap alone. (This project has a concrete example of "locked but caused by grasping": a MuJoCo physics fix shifted the resting pose of the bowl on the ramekin in task 5, and one layout made SmolVLA fail 13/13—fully locked—because of grasping, not wrong target selection. See "Version sensitivity.")

One more caveat: overlap is only informative for policies that **re-sample actions each time.** π0.5 re-samples via flow matching, so the two runs are genuinely independent. OFT does not sample—its two runs are near-replays (see "Limitations")—so perfect overlap is trivially expected and tells you neither "near-certain" nor anything about mechanism. This signal applies only to stochastic policies (SmolVLA, π0.5).

### SmolVLA: contact geometry throughout, zero referent confusion

- **Every reviewed failure is contact geometry, but coverage is limited.** Of 63 total failures, 12 have been video-reviewed: all 9 from task 9 (8 grasp failures + 1 off-center lift that missed the plate on placement; review log: [`failure_videos/smolvla_t9_mj327/`](failure_videos/smolvla_t9_mj327/ANNOTATION_zh.md)), plus 3 from task 5 (e0/e1/e9: 2 never lifted the bowl, 1 placed too far and landed on the plate edge; task 5 has 10 failures, 7 unrecorded). Within this reviewed set, **not a single instance of picking the wrong bowl.** (Under broken-premise physics, a causal chain for "grasp-side bias" was established for that era's task 5 failures via data intervention; that conclusion is scoped to that scenario and has not been re-verified under healthy physics. See [attribution report](attribution_framework_zh.md).)
- **No systematic referent confusion.** π0.5-style systematic confusion would depress a task score into the 80% range (π0.5 t7: 40/50). SmolVLA scores 48/50 on task 7—tied for its best—ruling out any systematic pattern. All task 9 failures were reviewed with zero confusion. Isolated cases in unreviewed failures cannot be excluded.

### OFT: mostly contact geometry, plus one isolated case

- **Coverage:** Every failure across both arms × both runs was video-reviewed—t7 and t9 in full, plus 18 layouts / 31 segments of scattered failures in other tasks. (Review logs: [`oft_t7/`](failure_videos/oft_t7/ANNOTATION_zh.md), [`oft_t9/`](failure_videos/oft_t9/ANNOTATION_zh.md), [`oft_rest/`](failure_videos/oft_rest/ANNOTATION_zh.md).)
- **Contact geometry is the dominant mechanism.** Among the 17 non-confused layouts in the scattered failures: 11 grasp failures, 6 in-transit drops, 0 placement errors. The "momentary jaw release during transport" pattern first noted in t7 extends to t2/t3/t6/t8.
- **One isolated referent confusion.** The combined checkpoint, on layout #22 of task 5, heads straight for the distractor bowl sitting on the cookie box in both runs—never approaching the target bowl. It does not even grasp the distractor; it pokes and knocks it off. The specialist checkpoint selects the correct target on the same layout and succeeds. This is not "aimed at the right bowl but missed by millimeters"—it is "decided on the wrong bowl before acting": an error in grounding (mapping the instruction noun to a specific scene object), a discrete either-or decision, not a precision issue. Single layout (1/500); not generalized. A symmetry note: the confused distractor here shares a "bowl sitting on a pedestal" geometry (ramekin ↔ cookie box) with π0.5's stove ↔ cabinet confusion—structurally analogous.
- **Two close-up observations.** (1) In task 7 cross-run flip pairs, the successful and failed runs exhibit the **same behavior**: the jaws briefly open during transport. The only difference is whether the opening persists long enough to drop the bowl. Action logs confirm the jaw command is "close" throughout—the opening is load-induced physics slippage, not a chunk-switching artifact. The success/failure boundary passes through the *interior* of a single behavior and is discretized by the success predicate. (2) In task 5 flip pairs, both runs execute a **dual-bowl co-grasp**—clamping the bowl and the ramekin together and lifting the entire stack. This sub-pattern was first named on 0.45B SmolVLA; here it appears at 7.5B under healthy physics. The successful run scores only because the ramekin accidentally falls off mid-transport.

### Cross-model view: failure mechanisms shift with capacity; they don't all improve monotonically

| Mechanism \ Model | SmolVLA 0.45B | π0.5 ~3.3B | OFT 7.5B |
|---|---|---|---|
| **Contact geometry** (grasp / transport / placement) | All reviewed failures belong here (12/63 covered) | Low rate (t5: 45/50) | Primary source, but rare (no task below 46/50) |
| **Referent confusion** | **Not observed** (t9 fully reviewed, zero confusion; t7 48/50 rules out systematic) | **Systematic paired confusion** (stove↔cabinet, 10+ layouts in t7/t9, locked under perturbation) | Single-layout isolate (combined arm, t5 layout #22) |

Three readings:

1. **The contact-geometry family (off-center grasp → rim landing, dual-bowl co-grasp) has video evidence at all three capacity points.** Capacity reduces incidence but does not eliminate the mechanism.
2. **Systematic referent confusion appears only in π0.5**: paired confusion between two adjacent elevated surfaces, absent in both the smaller and the larger model. This is not a monotonic "bigger is better" curve. Whether it should be attributed to capacity, the pretrained backbone, or the missing language-conditioned reasoning path in lerobot's port (see "Limitations") cannot be disentangled from three models, each with one checkpoint.
3. OFT at 7.5B covers both mechanism families, but covering ≠ eliminating: the isolated confusion and contact-margin failures persist.

## Finding: a MuJoCo bugfix that silently rewrites the benchmark

The largest single finding outside the main table: **LIBERO-Spatial's per-task numbers are highly sensitive to MuJoCo physics version, and the mechanism traces to task 5's initial-state premise.** During development we collected a full set of numbers under MuJoCo 3.8.1; after identifying this issue, everything was re-measured under a single physics version (the current main table). The 3.8.1 data does not serve as a leaderboard number; it is preserved here as the complete record of an accidental stress test. (Full case file: [attribution report](attribution_framework_zh.md) appendix and the companion project vla-rl-post-training findings.)

**The incident chain, in three acts:**

1. **The buried debt: initial-state semantics outsourced to the physics engine.** LIBERO's official task 5 `init_states` does not store the bowl's resting position. It stores the bowl **suspended ~11 cm above the ramekin**. On `env.reset()`, three things happen: the simulator restores this mid-air state; 10 no-op control steps run (~0.5 s sim time), letting the bowl fall, collide with the ramekin, slide, and settle; the first observation is then captured and handed to the policy. Recording starts from this frame (OFT's stack likewise waits for settling), so the entire drop-and-settle sequence completes before the first frame—**the landing never appears on camera.** Whether the bowl sits squarely on the ramekin or slides off and leans against it is entirely determined by how the physics engine resolves that 0.5 s, which is exactly what a version change can alter.

2. **The detonation: a correct bugfix.** MuJoCo 3.4.0 (2025-12-05) fixed the box–box collision distance calculation (bowl = 40 boxes, ramekin = 25 boxes; the settling is pure box–box contact). A GPU-free, policy-free physics probe (`set_state` + settle, seconds per version) across six versions: 3.2.7 / 3.3.0 / 3.3.7 all settle normally (1.45 cm slide, ~16° tilt). **3.4.0 / 3.6.0 / 3.8.1 all slide off and lean** (3.72 cm, 10.5°). Within each version, all 50 layouts produce bit-identical settling. **The boundary is exactly the 3.4.0 release.** The healthy settling relied on the old box–box bug; 3.4.0 is the correct fix; upstream will not revert. Pinning the old version is a stopgap. Upstream corroboration: LIBERO #141 (same slide-off under 3.6.0), SimVLA #14 (score gap between 3.1.0 and 3.9.0).

3. **Localization: same-physics A/B test.** Holding physics at 3.8.1 and replacing only task 5's init states with freshly generated pre-settled versions (healthy resting pose, verified to produce 0.00 cm additional slide under 3.8.1): SmolVLA task 5 goes from 28% → **84%**; OFT-GRPO (companion project, different eval stack, same A/B design) task 5 goes from 6/50 → **49/50**, matching the healthy-physics baseline, with all other nine tasks within noise. **The version effect on task 5 is 100% carried by the broken initial premise; the physics difference during policy execution is negligible. This holds across eval stacks (lerobot / RLinf) and across policies (SmolVLA / OFT).** (The 84% and 49/50 do not enter the main table—non-standard init, used only as attribution evidence.)

**The version effect is policy-dependent—what it really measures is robustness to a broken premise:**

| Model | @3.8.1 (broken premise) | @3.2.7 (healthy, main table) | Task 5 delta |
|---|---|---|---|
| SmolVLA 0.45B | 83.8 (task 5: 28%) | 87.4 (task 5: 80%) | +26 eps |
| π0.5 ~3.3B | 93.8 (task 5: 86%) | 94.4 (task 5: 90%) | +2 eps |
| OFT-GRPO 7.5B (companion project, greedy) | 85.0 (task 5: 12%) | 92.8 (task 5: 98%) | +43 eps |

Composition of the SmolVLA delta: total +18 episodes = task 5 +26 plus the other nine tasks net −8 (t0 −5, t4 −3, t3 −2, t1 −1, t2 +1, t8 +2, t6/t7/t9 unchanged)—the latter scattered, mixed in sign, within the stack's run-to-run noise band. π0.5's other nine tasks: net +1.

π0.5 receives the **same broken init** under 3.8.1 (frame-0 extraction confirms the bowl has already slid off) yet still scores 43/50—its "immunity" is policy robustness, not absence of the perturbation. The same broken-premise question opens a **58 pp gap** between SmolVLA and π0.5 (28% vs 86%)—the version artifact accidentally constitutes a stress test that the official protocol does not cover.

**An overturned conclusion: the "hardest layout."** Under broken-premise physics, layout #3 of task 5 was identified as the single hardest known layout (SmolVLA: 0/13 across all historical runs). Same layout, pinned, independently re-run 10 times per model with different policy seeds, under healthy physics:

| Model | @3.8.1 (broken premise) | @3.2.7 (healthy) |
|---|---|---|
| SmolVLA | 0/13 (all historical runs) | **9/10** |
| π0.5 | 5/10 | **10/10** |
| OFT spec. / comb. | — (not tested) | 1/1 each (n=1, weak perturbation; see below) |

**Under healthy physics it is not hard at all**—even the 0.45B model scores 9/10. The supposed "toxicity" was entirely the broken premise hitting this layout hardest; 0/13 and 5/10 measured performance under a corrupted premise, not layout difficulty.

**How to read pinned-layout re-run numbers.** All are hit rates; the differences lie in perturbation source, strength, and sample size. SmolVLA and π0.5 sample actions via flow matching—**strong, seeded perturbation** by design; *x*/10 is a hit rate over ten independent re-draws. OFT's L1 regression head does not sample, but closed-loop execution carries unseeded numerical noise from the rendering pipeline—**weak perturbation** with a low flip rate (see "Limitations"). OFT's 1/1 is a single sample under weak perturbation, not a definitive verdict. The two number types have different evidential weight—a protocol lesson for cross-model pinned-layout probes.

**Three implications for benchmark methodology:**

1. Per-task numbers are inherently sensitive to the "protocol fingerprint" (physics version, init-state semantics, reset details). Cross-protocol, cross-version number comparisons must align these first.
2. Ceiling-style version pins (e.g. hf-libero's `<3.9.0`) prevent API breakage but not behavioral drift. Benchmarks with physics-engine dependencies need **behavioral regression tests**—the settling probe from this project can serve as a CI check.
3. The root fix is self-sufficient data: store the post-settled state directly. Re-generation scripts and pre-settled init files are ready; upstream issues filed: [lerobot#4390](https://github.com/huggingface/lerobot/issues/4390), with follow-ups on LIBERO#141 / RLinf#1460.

## Limitations and open questions

- Main-table numbers are single runs. Run-to-run variation reference: π0.5's four worst tasks scored 173/200 and 171/200 in two same-stack runs under 3.8.1—a 2-episode spread (neither run is the main-table run). SmolVLA and π0.5 totals carry action-sampling noise on top of the CIs shown.
- **OFT is not deterministic.** Same weights, same config (md5-verified), full re-run: specialist 487→490 (same GPU, 7 episodes flip outcome), combined 487→489 (6 flips; re-run on a different card of the same model, GPU-as-factor not isolated). Run-to-run differences reported as observed (3/500 and 2/500; n=2 is not enough to estimate variance). Root-cause evidence: the policy forward pass is **bit-stable** under fixed input (10× in-process and cross-process hash checks all match), so the randomness is not in the policy head and `torch` determinism flags cannot reach it. A fork probe—two processes each running all 50 episodes of t7 while hashing sim state, policy input, both cameras, and actions at every step—shows 50/50 episodes diverge, and the first divergence is **always in the camera image** (wrist camera first 45/50, main camera first 5/50). Images diverge while sim state is still bit-identical. Conclusion: the random source is EGL off-screen rendering, which is not bit-reproducible across processes; pixel noise is global, but only marginal layouts get their outcome flipped. lerobot stack's same-seed irreproducibility is likely the same root cause (same MuJoCo + off-screen rendering pipeline) and could be reported upstream. A separate **cross-episode positional effect** exists: the same episode run in isolation vs within a 500-episode sequence can systematically differ in outcome (t7 episode 14: 4/4 fail in sequence, 2/2 succeed in isolation), traced to MuJoCo's warmstart cache and random-number stream position. Full case file: [`oft_nondeterminism_case_zh.md`](oft_nondeterminism_case_zh.md).
- OFT combined has only been evaluated on Spatial so far.
- **We evaluate π0.5 in pure action mode.** The lerobot port has no text-generation path: on load, `lm_head.weight` is remapped to `embed_tokens.weight` (`modeling_pi05.py:1111`); the module's only output projection is `action_out_proj`. The paper's hierarchical inference—decode a high-level subtask as language, then condition the action expert on it—is not ported and is out of scope here. Impact on attribution: the referent decision behind task-7 failures has **no observable intermediate** anywhere in this stack; the mechanism had to be established behaviorally (video + failure-set overlap) rather than by reading an internal variable.
- The two eval stacks differ beyond inference configs (policy-step budgets 280 vs 220, env library versions). We kept each model's official stack by design, disclosing rather than normalizing. MuJoCo version is the one difference proven to be a material variable; it is singled out as a finding, and all four rows are aligned to a single version (see that section).
- `failure_videos/ckpt030000_n1/` (archive from an earlier run) was produced in a now-deprecated environment whose exact MuJoCo version is unknown; not used as evidence.
- Object / Goal / Long suites and GR00T are not yet included.

## Reproduction

- SmolVLA / π0.5: `lerobot-eval` (lerobot 0.6.0) with the headline-table configs. ⚠️ Reproducing the main table requires `pip install mujoco==3.2.7` (3.4.0+ breaks the task 5 initial premise; see version-sensitivity section). Full-episode video: `scripts/lerobot_eval_fullvideo.py` (`EVAL_RENDER_ALL_N` knob).
- Pinned-layout re-runs: `scripts/attribution_probe.py --mode rollout --task-id 5 --init-index 3 --model-path <ckpt> --n-action-steps <official>`.
- Version-sensitivity probes (no policy, no GPU, runs in seconds): `scripts/mj_settle_probe.py` (six-version settling matrix; boundary at 3.4.0); `scripts/mj_reseat_probe.py` (re-generate pre-settled init states + settling stability verification).
- OFT: official `openvla-oft` repo eval script, protocol unmodified. Note: reproducing a single episode in isolation is not equivalent to its result within a 500-episode sequence (see "Limitations"). Re-run stability = run the official script twice, diff per-episode outcomes. Bit-stability probe: `scripts/oft_bitstability_probe.py`. Fork probe: `scripts/oft_divergence_probe.py` + analyzer `scripts/oft_div_analyze.py`. Single-episode action log: `scripts/oft_action_trace.py`.
