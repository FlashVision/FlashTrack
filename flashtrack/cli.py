"""FlashTrack CLI — command-line interface for training, tracking, validation, and export."""

import argparse
import sys


def _colored(text, color):
    """Simple ANSI color helper."""
    colors = {"green": "\033[92m", "blue": "\033[94m", "yellow": "\033[93m", "red": "\033[91m", "bold": "\033[1m"}
    return f"{colors.get(color, '')}{text}\033[0m"


def _print_banner():
    print(_colored("FlashTrack", "bold") + f" v{_get_version()}")
    print(_colored("Ultra-lightweight real-time multi-object tracking", "blue"))
    print()


def _get_version():
    from flashtrack import __version__
    return __version__


def cmd_version(args):
    """Print version info."""
    _print_banner()


def cmd_settings(args):
    """Print system settings and environment info."""
    import torch
    import platform
    import numpy as np

    _print_banner()
    print(_colored("System", "bold"))
    print(f"  Python:      {platform.python_version()}")
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Machine:     {platform.machine()}")
    print()
    print(_colored("Dependencies", "bold"))
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  CUDA:        {torch.version.cuda or 'Not available'}")
    print(f"  cuDNN:       {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    try:
        import scipy
        print(f"  SciPy:       {scipy.__version__}")
    except ImportError:
        print("  SciPy:       Not installed")
    print()
    print(_colored("Hardware", "bold"))
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM:        {mem:.1f} GB")
    else:
        print("  GPU:         None (CPU only)")
    print(f"  CPU cores:   {__import__('os').cpu_count()}")


def cmd_check(args):
    """Verify installation — imports, GPU, and basic inference."""
    _print_banner()
    errors = []

    print(_colored("Checking installation...", "bold"))
    print()

    try:
        import flashtrack  # noqa: F401
        print(f"  {_colored('✓', 'green')} flashtrack package")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} flashtrack package: {e}")
        errors.append(str(e))

    try:
        from flashtrack.engine import Trainer, Predictor, Exporter, Validator  # noqa: F401
        print(f"  {_colored('✓', 'green')} engine (Trainer, Predictor, Exporter, Validator)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} engine: {e}")
        errors.append(str(e))

    try:
        from flashtrack.models import ByteTracker, SORTTracker, DeepSORTTracker  # noqa: F401
        print(f"  {_colored('✓', 'green')} trackers (ByteTracker, SORT, DeepSORT)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} trackers: {e}")
        errors.append(str(e))

    try:
        from flashtrack.analytics import Benchmark, Profiler  # noqa: F401
        print(f"  {_colored('✓', 'green')} analytics (Benchmark, Profiler)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} analytics: {e}")
        errors.append(str(e))

    try:
        import torch
        from flashtrack.cfg import get_config
        from flashtrack.models import build_model
        cfg = get_config(model_size="m", input_size=(128, 64), num_ids=10)
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 3, 128, 64))
        print(f"  {_colored('✓', 'green')} model forward pass (FlashTrack-m, 128x64)")
    except Exception as e:
        print(f"  {_colored('✗', 'red')} model forward pass: {e}")
        errors.append(str(e))

    import torch
    if torch.cuda.is_available():
        print(f"  {_colored('✓', 'green')} CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  {_colored('⚠', 'yellow')} No CUDA GPU (training will be slow)")

    print()
    if errors:
        print(_colored(f"✗ {len(errors)} check(s) failed", "red"))
        sys.exit(1)
    else:
        print(_colored("✓ All checks passed! FlashTrack is ready.", "green"))


def cmd_train(args):
    """Train a FlashTrack ReID model."""
    from flashtrack.engine.trainer import Trainer

    if args.config:
        from flashtrack.cfg import load_yaml_config
        cfg = load_yaml_config(args.config)
        print(f"{_colored('Config:', 'bold')} {args.config}")
        trainer = Trainer(config=cfg, device=args.device)
    else:
        if not args.train_data:
            print(_colored("Error:", "red") + " --train-data is required (or use --config)")
            sys.exit(1)
        kwargs = {
            "model_size": args.model_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "train_data": args.train_data,
            "val_data": args.val_data,
            "save_dir": args.save_dir,
        }
        if args.lora:
            kwargs["lora"] = True
        if args.qlora:
            kwargs["qlora"] = True
        if args.amp:
            kwargs["amp"] = True
        if args.lr:
            kwargs["lr"] = args.lr
        if args.workers is not None:
            kwargs["workers"] = args.workers
        trainer = Trainer(**kwargs)

    trainer.train()


def cmd_track(args):
    """Run tracking on a video."""
    from flashtrack.engine.predictor import Predictor

    predictor = Predictor(
        model_path=args.model,
        device=args.device,
        tracker_type=args.tracker,
        track_thresh=args.thresh,
    )

    predictor.track_video(args.source, output_dir=args.output, show=args.show)


def cmd_val(args):
    """Validate tracking model."""
    from flashtrack.engine.validator import Validator
    validator = Validator(
        model_path=args.model,
        val_data=args.val_data,
        device=args.device,
    )
    validator.validate()


def cmd_export(args):
    """Export model to ONNX."""
    from flashtrack.engine.exporter import Exporter
    exporter = Exporter(model_path=args.model)
    path = exporter.export(output=args.output, simplify=args.simplify)
    print(f"\n{_colored('✓', 'green')} Exported: {path}")


def main():
    parser = argparse.ArgumentParser(
        prog="flashtrack",
        description="FlashTrack: Ultra-lightweight real-time multi-object tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashtrack check                              Verify installation
  flashtrack train --train-data data/MOT17/train
  flashtrack track --model best.pth --source video.mp4
  flashtrack export --model best.pth --output reid.onnx --simplify

Documentation: https://github.com/FlashVision/FlashTrack
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("version", help="Show version info")
    subparsers.add_parser("settings", help="Show system settings (Python, PyTorch, CUDA, GPU)")
    subparsers.add_parser("check", help="Verify installation and run health check")

    # train
    train_p = subparsers.add_parser("train", help="Train a FlashTrack ReID model")
    train_p.add_argument("--config", default=None, help="Path to YAML config")
    train_p.add_argument("--model-size", default="m", choices=["m-0.5x", "m", "m-1.5x"])
    train_p.add_argument("--epochs", type=int, default=120)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=None)
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--train-data", default=None, help="Path to MOT train data")
    train_p.add_argument("--val-data", default=None, help="Path to MOT val data")
    train_p.add_argument("--save-dir", default="workspace/tracking")
    train_p.add_argument("--workers", type=int, default=None)
    train_p.add_argument("--lora", action="store_true", help="Enable LoRA")
    train_p.add_argument("--qlora", action="store_true", help="Enable QLoRA")
    train_p.add_argument("--amp", action="store_true", help="Enable mixed precision")

    # track
    track_p = subparsers.add_parser("track", help="Run tracking on video")
    track_p.add_argument("--model", default=None, help="Path to ReID model checkpoint")
    track_p.add_argument("--source", required=True, help="Video path")
    track_p.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "sort", "deepsort"])
    track_p.add_argument("--thresh", type=float, default=0.5, help="Track confidence threshold")
    track_p.add_argument("--device", default="cuda")
    track_p.add_argument("--output", default="output", help="Output directory")
    track_p.add_argument("--show", action="store_true", help="Display video")

    # val
    val_p = subparsers.add_parser("val", help="Validate tracking model")
    val_p.add_argument("--model", required=True, help="Path to checkpoint")
    val_p.add_argument("--val-data", required=True, help="Path to validation data")
    val_p.add_argument("--device", default="cuda")

    # export
    exp_p = subparsers.add_parser("export", help="Export model to ONNX")
    exp_p.add_argument("--model", required=True, help="Path to checkpoint")
    exp_p.add_argument("--output", default="reid_model.onnx")
    exp_p.add_argument("--simplify", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        _print_banner()
        parser.print_help()
        sys.exit(0)

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "train": cmd_train,
        "track": cmd_track,
        "val": cmd_val,
        "export": cmd_export,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
