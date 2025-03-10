"""学習率スケジューラーモジュール."""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR(_LRScheduler):
    """Cosine Annealing with Warm-up学習率スケジューラー.

    学習率をmax_epochsに至るまでコサイン曲線に沿ってスケジュールする。
    epoch 0からwarmup_epochsまでの学習曲線は線形warmupがかかる。

    参考: https://pytorch-lightning-bolts.readthedocs.io/en/stable/schedulers/warmup_cosine_annealing.html
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.00001,
        eta_min: float = 0.00001,
        last_epoch: int = -1,
    ) -> None:
        """初期化.

        Args:
            optimizer: 最適化手法インスタンス
            warmup_epochs: linear warmupを行うepoch数
            max_epochs: cosine曲線の終了に用いる学習のepoch数
            warmup_start_lr: linear warmup 0 epoch目の学習率
            eta_min: cosine曲線の下限
            last_epoch: cosine曲線の位相オフセット
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """現在のepochにおける学習率を取得.

        Returns:
            現在のepochにおける学習率のリスト
        """
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)

        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
