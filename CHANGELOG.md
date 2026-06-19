# Changelog

All notable changes to FlashTrack will be documented in this file.

## [1.0.0] — 2026-06-19

### Added
- **Package structure** — `pip install` from GitHub or PyPI
- **CLI** — `flashtrack train`, `track`, `val`, `export`, `check`, `settings`, `version`
- **Python API** — `Trainer`, `Predictor`, `Exporter`, `Validator`
- **ReID Model** — FlashTrack-m-0.5x, FlashTrack-m, FlashTrack-m-1.5x
- **LoRA fine-tuning** — 6 variants (standard, dora, lora_plus, adalora, ortho, lora_fa)
- **QLoRA** — INT8/NF4 quantized base weights + LoRA
- **Knowledge Distillation** — teacher-student ReID training
- **Trackers** — ByteTracker, SORTTracker, DeepSORTTracker
- **Losses** — Triplet loss with hard mining, cross-entropy ID classification
- **Metrics** — MOTA, MOTP, IDF1, ID switches, track fragmentation
- **Analytics** — Benchmark, Profiler
- **ONNX export** — with simplification support
- **Mixed precision** — AMP (FP16) training
- **CI/CD** — GitHub Actions (lint + test on Python 3.9-3.12, auto-publish to PyPI)
- **Examples** — 5 runnable example scripts

### Architecture
- ShuffleNetV2 backbone (0.5x, 1.0x, 1.5x)
- Lightweight CNN feature encoder
- ReID head with BN neck + optional classification branch
- Kalman filter for state estimation
- Hungarian algorithm for assignment
