# smolvla_base on LIBERO, zero-shot: N/A (structural mismatch)

**Date**: 2026-07-14 · **Verdict**: zero-shot evaluation is *undefined*, not merely low-scoring.

`lerobot/smolvla_base` was pretrained exclusively on SO-100 community data. Its I/O
interface is dimensionally incompatible with the LIBERO environment (Franka Panda),
so there is no honest way to roll it out without retraining the I/O projections —
which would no longer be "pre-finetune":

| Interface | smolvla_base (SO-100 pretrain) | LIBERO env (Panda) |
| --------- | ------------------------------ | ------------------- |
| Cameras   | 3 (`camera1..3`, 256×256)      | 2 (agentview + wrist, 256×256) |
| Proprioceptive state | 6-dim (joint positions) | 8-dim (EEF pose + gripper) |
| Action    | 6-dim (joint commands)         | 7-dim (6-DoF EEF delta + gripper) |

Evidence: `input_features` / `output_features` in the checkpoint's `config.json`
versus the env spec in the [official LIBERO docs](https://huggingface.co/docs/lerobot/main/en/libero)
(`lerobot-eval` fails fast with a feature-mismatch error before any rollout).

## Implications for the leaderboard

- The pre-finetune baseline row for SmolVLA is reported as **N/A (structural
  mismatch)** rather than 0%.
- The finetuning success-rate curve starts near zero at low step counts; that
  early point serves as the denominator for improvement claims.
- Contrast: OpenVLA-class models pretrained on multi-embodiment OXE data with a
  7-dim EEF action space can be rolled out on LIBERO zero-shot. The boundary of
  "generalist" VLA claims is exactly this: embodiment coverage of the
  pretraining mixture.
