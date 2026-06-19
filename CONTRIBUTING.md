# Contributing to FlashTrack

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/FlashVision/FlashTrack.git
cd FlashTrack
pip install -e ".[dev,all]"
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run lint: `ruff check flashtrack/`
4. Run tests: `flashtrack check`
5. Commit and push
6. Open a Pull Request

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting (line length: 120)
- Type hints are encouraged
- Docstrings for all public functions (Google style)
- No hardcoded file paths — use relative or configurable paths

## Adding a New Tracker

1. Create `flashtrack/models/your_tracker.py`
2. Implement `update(detections)` → `np.ndarray` of shape `(N, 7)`
3. Implement `reset()`
4. Add to `flashtrack/models/__init__.py`
5. Register in `flashtrack/registry.py`

## Adding a New Loss

1. Create `flashtrack/losses/your_loss.py`
2. Implement as `nn.Module` with `forward()` returning scalar loss
3. Add to `flashtrack/losses/__init__.py`

## Commit Messages

Use clear, descriptive messages:
- `Add DeepSORT tracker with ReID matching`
- `Fix Kalman filter aspect ratio handling`
- `Update README with tracking examples`

## Reporting Issues

- Use GitHub Issues
- Include: Python version, PyTorch version, GPU info, error traceback
- Run `flashtrack settings` and paste the output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
