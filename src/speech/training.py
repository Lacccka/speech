"""Training utilities for speech recognition models."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from .config import TrainingConfig
from .logging import get_logger


class BatchProtocol(Protocol):
    """Protocol describing the batches expected by :class:`Trainer`."""

    def __iter__(self) -> Iterable[torch.Tensor]:  # pragma: no cover - typing helper
        ...


@dataclass(slots=True)
class TrainState:
    """Dataclass storing mutable training state."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")


class Trainer:
    """High-level training loop with validation and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: Iterable[BatchProtocol],
        *,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        val_loader: Optional[Iterable[BatchProtocol]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.state = TrainState()

        self.scaler = GradScaler(enabled=config.mixed_precision)
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _forward_step(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        autocast_dtype = getattr(torch, self.config.mixed_precision_dtype, torch.float16)
        with autocast(enabled=self.scaler.is_enabled(), dtype=autocast_dtype):
            logits, output_lengths = self.model(features, feature_lengths)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = self.criterion(
                log_probs.transpose(0, 1), targets, output_lengths, target_lengths
            )
        return loss, output_lengths

    def _prepare_batch(
        self, batch: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) != 4:
            raise ValueError(
                "Each batch must contain (features, feature_lengths, targets, target_lengths)"
            )
        features, feature_lengths, targets, target_lengths = batch
        features = features.to(self.device)
        feature_lengths = feature_lengths.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)
        return features, feature_lengths, targets, target_lengths

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(self.train_loader, start=1):
            features, feature_lengths, targets, target_lengths = self._prepare_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss, _ = self._forward_step(features, feature_lengths, targets, target_lengths)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.config.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.optimizer.step()

            total_loss += loss.item()
            self.state.global_step += 1
            batch_count += 1

            if batch_idx % self.config.log_interval == 0:
                avg_loss = total_loss / batch_count
                self.logger.info(
                    "Epoch %d | Step %d | Avg loss %.4f",
                    self.state.epoch + 1,
                    batch_idx,
                    avg_loss,
                )

        if batch_count == 0:
            return 0.0

        return total_loss / batch_count

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {"loss": float("nan")}

        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch in self.val_loader:
            features, feature_lengths, targets, target_lengths = self._prepare_batch(batch)
            loss, _ = self._forward_step(features, feature_lengths, targets, target_lengths)
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        return {"loss": avg_loss}

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool,
    ) -> Path:
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "metrics": metrics,
        }
        if self.scheduler is not None:
            payload["scheduler_state"] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(payload, checkpoint_path)
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(payload, best_path)
            return best_path
        return checkpoint_path

    def _step_scheduler(self, metric: float) -> None:
        if self.scheduler is None:
            return

        try:
            signature = inspect.signature(self.scheduler.step)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            self.scheduler.step()
            return

        if len(signature.parameters) == 0:
            self.scheduler.step()
        else:
            self.scheduler.step(metric)

    def fit(self) -> None:
        """Execute the configured training loop."""

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            val_metric = val_metrics.get(self.config.metric_name, train_loss)

            improved = val_metric < self.state.best_metric
            if improved:
                self.state.best_metric = val_metric

            if not self.config.save_best_only or improved:
                checkpoint_path = self._save_checkpoint(epoch + 1, val_metrics, improved)
                self.logger.info("Saved checkpoint to %s", checkpoint_path)

            self._step_scheduler(val_metric)

            self.logger.info(
                "Epoch %d completed | train_loss=%.4f | %s=%.4f",
                epoch + 1,
                train_loss,
                self.config.metric_name,
                val_metric,
            )


__all__ = ["Trainer", "TrainState"]

