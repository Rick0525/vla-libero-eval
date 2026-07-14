# VLA Eval Harness

[中文版 / Chinese version](README_zh.md)

A unified evaluation harness for open-source Vision-Language-Action (VLA) models on LIBERO — standardized leaderboard, failure attribution (perception / planning / control), trajectory visualization, and one-command reproduction.

> 🚧 **Work in progress** (started 2026-07-14). Models planned: SmolVLA, OpenVLA-OFT, π0/π0.5, GR00T. First milestone: pre-finetune SmolVLA baseline on LIBERO-Spatial.

## Why another eval?

Published VLA numbers are rarely comparable: episode counts, seeds, max steps, and camera setups differ across papers, and community reproductions can diverge wildly from reported results. This harness evaluates every model under one fixed protocol, then goes beyond the success-rate scalar with per-episode failure attribution.

## Structure

```
scripts/   # evaluation & reproduction scripts
results/   # leaderboard tables and curves (small files only)
```

More to come as the project grows.

## License

MIT
