"""Example: Fine-tune a ReID model with LoRA."""

from flashtrack import Trainer


def main():
    trainer = Trainer(
        model_size="m",
        epochs=60,
        batch_size=64,
        lr=0.001,
        train_data="data/MOT17/train",
        lora=True,
        lora_rank=8,
        lora_variant="standard",
        amp=True,
        save_dir="workspace/reid_lora",
    )

    results = trainer.train()
    print(f"Best loss: {results['best_loss']:.4f}")


if __name__ == "__main__":
    main()
