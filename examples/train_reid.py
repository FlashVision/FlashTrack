"""Example: Train a ReID model on MOT17 data."""

from flashtrack import Trainer


def main():
    trainer = Trainer(
        model_size="m",
        epochs=120,
        batch_size=64,
        lr=0.0003,
        train_data="data/MOT17/train",
        val_data="data/MOT17/val",
        amp=True,
        save_dir="workspace/reid_training",
    )

    results = trainer.train()
    print(f"Best loss: {results['best_loss']:.4f}")


if __name__ == "__main__":
    main()
