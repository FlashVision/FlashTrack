"""FlashTrack Trainer — ReID feature learning with triplet + classification loss."""

import os
import copy
import math
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from flashtrack.cfg import get_config
from flashtrack.models.tracker import FlashTracker
from flashtrack.models.lora import apply_lora, apply_qlora, merge_lora_weights, get_lora_state_dict
from flashtrack.data import create_dataloader
from flashtrack.losses.triplet_loss import TripletLoss
from flashtrack.losses.classification_loss import ClassificationLoss
from flashtrack.utils import (
    save_checkpoint, load_checkpoint, save_weights_only, save_inference_weights,
    setup_logger, AverageMeter,
)
from flashtrack.utils.torchtune_optim import (
    apply_activation_checkpointing,
    ActivationOffloadHook,
    create_optimizer,
    compile_model as torchtune_compile,
)

logger = logging.getLogger(__name__)


class ModelEMA:
    """Exponential Moving Average of model weights with adaptive decay warmup."""

    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup: int = 2000):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.target_decay = decay
        self.warmup = warmup
        self.num_updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @property
    def decay(self):
        return min(self.target_decay,
                   (1 + self.num_updates) / (self.warmup + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = self.decay
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self):
        return {
            "ema_state": self.ema.state_dict(),
            "target_decay": self.target_decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict):
        self.ema.load_state_dict(state["ema_state"], strict=False)
        self.target_decay = state.get("target_decay", self.target_decay)
        self.warmup = state.get("warmup", self.warmup)
        self.num_updates = state.get("num_updates", 0)


MODEL_SIZE_MAP = {
    "m": {"backbone": "1.0x", "encoder_channels": 256, "reid_dim": 128},
    "m-1.5x": {"backbone": "1.5x", "encoder_channels": 384, "reid_dim": 256},
    "m-0.5x": {"backbone": "0.5x", "encoder_channels": 128, "reid_dim": 128},
}


class Trainer:
    """High-level trainer for FlashTrack ReID model.

    Example::

        from flashtrack import Trainer

        trainer = Trainer(
            epochs=120,
            batch_size=64,
            model_size="m",
            train_dir="data/MOT17/train",
            lora=True,
            amp=True,
        )
        trainer.train()
    """

    def __init__(
        self,
        epochs: int = 120,
        batch_size: int = 64,
        lr: float = 0.0003,
        workers: int = 4,
        save_dir: str = "workspace/tracking_experiment",
        resume: Optional[str] = None,
        device: str = "cuda",
        warmup_epochs: int = 5,
        patience: int = 50,
        model_size: str = "m",
        input_size: tuple = (128, 64),
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        amp: bool = False,
        grad_accum: int = 1,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        optimizer_in_bwd: bool = False,
        use_8bit_optimizer: bool = False,
        compile: bool = False,
        lora: bool = False,
        lora_variant: str = "standard",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets: Optional[List[str]] = None,
        qlora: bool = False,
        qlora_dtype: str = "int8",
        triplet_weight: float = 1.0,
        classification_weight: float = 0.5,
        triplet_margin: float = 0.3,
        config: Any = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.workers = workers
        self.save_dir = save_dir
        self.resume = resume
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.model_size = model_size
        self.input_size = input_size
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.amp = amp
        self.grad_accum = max(1, grad_accum)
        self.activation_checkpointing = activation_checkpointing
        self.activation_offloading = activation_offloading
        self.optimizer_in_bwd = optimizer_in_bwd
        self.use_8bit_optimizer = use_8bit_optimizer
        self.compile = compile
        self.lora = lora
        self.lora_variant = lora_variant
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = lora_targets or ["backbone", "encoder"]
        self.qlora = qlora
        self.qlora_dtype = qlora_dtype
        self.triplet_weight = triplet_weight
        self.classification_weight = classification_weight
        self.triplet_margin = triplet_margin

        self._config = config or get_config(model_size=model_size, input_size=input_size)
        self._model_cfg = MODEL_SIZE_MAP.get(self.model_size, MODEL_SIZE_MAP["m"])

        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            if device not in ("cpu", ""):
                logger.warning("CUDA unavailable; falling back to CPU.")

        os.makedirs(self.save_dir, exist_ok=True)
        self._logger = setup_logger("FlashTrack", self.save_dir)

    def train(self) -> Dict[str, float]:
        """Run the full ReID training loop. Returns dict with best_rank1 and best_loss."""
        cfg = self._config

        train_dir = self.train_dir or cfg.data.train_images
        val_dir = self.val_dir or cfg.data.val_images

        self._logger.info("=" * 60)
        self._logger.info("FlashTrack ReID Training")
        self._logger.info("=" * 60)
        self._logger.info(f"Device: {self.device}")
        self._logger.info(f"Model: {self.model_size}, Input: {self.input_size}")
        self._logger.info(f"Epochs: {self.epochs}, Batch: {self.batch_size}, LR: {self.lr}")

        train_loader = create_dataloader(
            root_dir=train_dir,
            mode="reid",
            batch_size=self.batch_size,
            crop_size=self.input_size,
            num_workers=self.workers,
            is_train=True,
        )
        val_loader = create_dataloader(
            root_dir=val_dir,
            mode="reid",
            batch_size=self.batch_size,
            crop_size=self.input_size,
            num_workers=self.workers,
            is_train=False,
        )

        num_ids = train_loader.dataset.num_ids
        self._logger.info(f"Number of identities: {num_ids}")

        model = FlashTracker(
            backbone_size=self._model_cfg["backbone"],
            encoder_channels=self._model_cfg["encoder_channels"],
            reid_dim=self._model_cfg["reid_dim"],
            num_ids=num_ids,
            pretrained=True,
            input_size=self.input_size,
        ).to(self.device)

        model = self._apply_lora(model)

        triplet_loss = TripletLoss(margin=self.triplet_margin).to(self.device)
        cls_loss = ClassificationLoss(num_classes=num_ids, label_smooth=0.1).to(self.device)

        scaler = None
        if self.amp and self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            self._logger.info("AMP enabled")

        if self.activation_checkpointing:
            apply_activation_checkpointing(model, target_modules=["backbone", "encoder"])
        offload_hook = None
        if self.activation_offloading:
            offload_hook = ActivationOffloadHook()
            offload_hook.register(model)
        if self.compile:
            model = torchtune_compile(model)

        optimizer = create_optimizer(
            model, lr=self.lr, weight_decay=cfg.train.weight_decay,
            use_8bit=self.use_8bit_optimizer, optimizer_in_bwd=self.optimizer_in_bwd,
        )

        eta_min = 1e-6
        eta_min_factor = eta_min / self.lr

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_factor + (1.0 - eta_min_factor) * cosine

        scheduler = None
        if not self.optimizer_in_bwd:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        ema = ModelEMA(model, decay=0.9998, warmup=2000)

        start_epoch = 0
        best_loss = float("inf")
        best_rank1 = 0.0

        if self.resume:
            ckpt = load_checkpoint(model, self.resume, optimizer, scheduler, str(self.device))
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt.get("loss", float("inf"))
            self._logger.info(f"Resumed from epoch {start_epoch}")

        model_config = {
            "backbone_size": self._model_cfg["backbone"],
            "encoder_channels": self._model_cfg["encoder_channels"],
            "reid_dim": self._model_cfg["reid_dim"],
            "num_ids": num_ids,
            "input_size": self.input_size,
        }

        self._logger.info("\nStarting training...")
        epochs_no_improve = 0

        for epoch in range(start_epoch, self.epochs):
            if self.optimizer_in_bwd:
                lr_factor = lr_lambda(epoch)
                current_lr = self.lr * lr_factor
                optimizer.set_lr(current_lr)
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            self._logger.info(f"\nEpoch {epoch + 1}/{self.epochs} (lr={current_lr:.6f})")

            train_metrics = self._train_one_epoch(
                model, train_loader, optimizer, triplet_loss, cls_loss, ema, scaler
            )

            if (epoch + 1) % cfg.train.val_interval == 0:
                val_metrics = self._validate(ema.ema, val_loader)
                rank1 = val_metrics.get("rank1", 0.0)
                val_loss = val_metrics.get("loss", float("inf"))

                if val_loss < best_loss:
                    best_loss = val_loss

                if rank1 > best_rank1:
                    best_rank1 = rank1
                    epochs_no_improve = 0
                    save_checkpoint(
                        model, optimizer, epoch, val_loss,
                        os.path.join(self.save_dir, "checkpoint_best.pth"),
                        scheduler=scheduler, config=model_config,
                    )
                    save_inference_weights(
                        ema.ema,
                        os.path.join(self.save_dir, "model_best_inference.pth"),
                        config=model_config,
                    )
                    self._logger.info(f"  Best model saved (rank-1: {best_rank1:.4f})")
                else:
                    epochs_no_improve += cfg.train.val_interval

                if self.patience > 0 and epochs_no_improve >= self.patience:
                    self._logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            save_checkpoint(
                model, optimizer, epoch, train_metrics["loss"],
                os.path.join(self.save_dir, "checkpoint_last.pth"),
                scheduler=scheduler, config=model_config, ema=ema,
            )

            if scheduler is not None:
                scheduler.step()

        if self.lora or self.qlora:
            lora_path = os.path.join(self.save_dir, "lora_adapters.pth")
            torch.save(get_lora_state_dict(ema.ema), lora_path)
            merge_lora_weights(ema.ema)

        save_inference_weights(
            ema.ema,
            os.path.join(self.save_dir, "model_final_inference.pth"),
            config=model_config,
        )

        if offload_hook is not None:
            offload_hook.remove()

        self._logger.info("=" * 60)
        self._logger.info("Training Complete!")
        self._logger.info(f"Best Rank-1: {best_rank1:.4f}  |  Best Loss: {best_loss:.4f}")
        self._logger.info("=" * 60)

        return {"best_rank1": best_rank1, "best_loss": best_loss}

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        if self.qlora:
            model = apply_qlora(
                model, rank=self.lora_rank, alpha=self.lora_alpha,
                dropout=self.lora_dropout, target_modules=self.lora_targets,
                quant_dtype=self.qlora_dtype, variant=self.lora_variant,
            )
            self._logger.info(f"QLoRA applied (rank={self.lora_rank})")
        elif self.lora:
            model = apply_lora(
                model, rank=self.lora_rank, alpha=self.lora_alpha,
                dropout=self.lora_dropout, target_modules=self.lora_targets,
                variant=self.lora_variant,
            )
            self._logger.info(f"LoRA applied (rank={self.lora_rank})")
        return model

    def _train_one_epoch(self, model, dataloader, optimizer, triplet_loss, cls_loss, ema, scaler):
        model.train()
        use_amp = scaler is not None
        loss_meter = AverageMeter("Loss")
        triplet_meter = AverageMeter("Triplet")
        cls_meter = AverageMeter("ClsLoss")

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast(self.device.type, enabled=use_amp):
                output = model(images, return_logits=True)
                embeddings = output["embeddings"]

                t_loss = triplet_loss(embeddings, labels) * self.triplet_weight
                c_loss = torch.tensor(0.0, device=self.device)
                if "logits" in output:
                    c_loss = cls_loss(output["logits"], labels) * self.classification_weight

                loss = (t_loss + c_loss) / self.grad_accum

            if torch.isnan(loss):
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update(model)

            total_loss = (t_loss + c_loss).item()
            loss_meter.update(total_loss)
            triplet_meter.update(t_loss.item())
            cls_meter.update(c_loss.item())

            if (batch_idx + 1) % 20 == 0:
                self._logger.info(
                    f"  [{batch_idx+1}/{len(dataloader)}] "
                    f"Loss: {loss_meter.avg:.4f} (trip={triplet_meter.avg:.4f}, cls={cls_meter.avg:.4f})"
                )

        return {"loss": loss_meter.avg, "triplet": triplet_meter.avg, "cls": cls_meter.avg}

    @torch.no_grad()
    def _validate(self, model, dataloader):
        model.eval()
        all_embeddings = []
        all_labels = []

        for images, labels in dataloader:
            images = images.to(self.device)
            embeddings = model.extract(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # CMC evaluation (rank-1 accuracy via cosine similarity)
        sim = embeddings @ embeddings.T
        sim.fill_diagonal_(-1e9)

        _, topk = sim.topk(10, dim=1)
        correct = labels[topk] == labels.unsqueeze(1)

        rank1 = correct[:, 0].float().mean().item()
        rank5 = correct[:, :5].any(dim=1).float().mean().item()

        self._logger.info(f"  Val: Rank-1={rank1:.4f}, Rank-5={rank5:.4f}")
        model.train()

        return {"rank1": rank1, "rank5": rank5, "loss": 1.0 - rank1}
