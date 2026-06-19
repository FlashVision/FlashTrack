# Installation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for training)

## Install from Source

```bash
git clone https://github.com/FlashVision/FlashTrack.git
cd FlashTrack
pip install -e ".[all]"
```

## Install with Specific Extras

```bash
# Core only (tracking + training)
pip install -e .

# With ONNX export
pip install -e ".[export]"

# With analytics (matplotlib, pandas)
pip install -e ".[analytics]"

# Development (testing, linting)
pip install -e ".[dev]"
```

## Verify Installation

```bash
flashtrack check
```

This verifies:
- Package imports work
- Engine components load
- Tracker algorithms work
- Model forward pass succeeds
- GPU availability

## Docker

```bash
cd docker
docker compose up --build
```
