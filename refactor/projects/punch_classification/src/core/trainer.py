"""学習実行モジュール."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.train_config import TrainConfig
from utils.eval_utils import (
    calculate_metrics,
    calculate_metrics_with_conditions,
    calculate_prauc,
)


@dataclass
class TrainState:
    """学習の状態を保持するクラス."""

    best_eval_loss: float = float("inf")
    best_mean_pr_auc: float = 0.0
    train_losses: list[float] = None
    eval_losses: list[float] = None

    def __post_init__(self) -> None:
        """初期化後の処理."""
        if self.train_losses is None:
            self.train_losses = []
        if self.eval_losses is None:
            self.eval_losses = []


class Trainer:
    """学習を実行するクラス."""

    def __init__(
        self,
        config: TrainConfig,
        model: nn.Module,
        criterion: nn.Module | callable,
        optimizer: Optimizer,
        scheduler: _LRScheduler | None = None,
        weights_tensor: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        """初期化.

        Args:
            config: 学習設定
            model: モデル
            criterion: 損失関数
            optimizer: オプティマイザ
            scheduler: 学習率スケジューラ
            weights_tensor: CBLoss用の重み
            device: デバイス
        """
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weights_tensor = weights_tensor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = TrainState()

        os.makedirs(config.save_dir, exist_ok=True)

    def train_epoch(
        self, epoch: int, train_loader: DataLoader, smooth_alpha: float = 0.0
    ) -> float:
        """1エポックの学習を実行する.

        Args:
            epoch: 現在のエポック数
            train_loader: 学習用データローダー
            smooth_alpha: ラベルスムージングの係数

        Returns:
            平均損失値
        """
        self.model.train()
        running_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            if smooth_alpha > 0.0:
                smoothed_targets = (1.0 - smooth_alpha) * targets + (1.0 - smooth_alpha) / 43
            else:
                smoothed_targets = targets

            images = images.to(self.device)
            targets = targets.to(self.device)
            smoothed_targets = smoothed_targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            if self.config.loss_type == "bce":
                loss = self.criterion(outputs, smoothed_targets)
            elif self.config.loss_type == "cbloss":
                weights = self.weights_tensor.repeat(targets.shape[0], 1)
                weights = weights.sum(1).unsqueeze(1).repeat(1, 43)
                loss = self.criterion(outputs, smoothed_targets, weight=weights)
            else:  # cbloss_drw
                if epoch < self.config.num_epochs - 5:
                    loss = self.criterion(outputs, smoothed_targets)
                else:
                    weights = self.weights_tensor.repeat(targets.shape[0], 1) * targets
                    weights = weights.sum(1).unsqueeze(1).repeat(1, 43)
                    loss = self.criterion(outputs, smoothed_targets, weight=weights)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def evaluate(self, eval_loader: DataLoader) -> tuple[float, float, list[dict[str, float]]]:
        """評価を実行する.

        Args:
            eval_loader: 評価用データローダー

        Returns:
            平均損失値、平均PR-AUC、各閾値での評価指標のタプル
        """
        self.model.eval()
        eval_loss = 0.0
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for images, targets in tqdm(eval_loader, desc="Evaluating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)

                loss = self.criterion(outputs, targets)
                eval_loss += loss.item()

                preds_list.append(outputs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        metrics_list = []
        pr_aucs = {}

        for th in range(5, 100, 5):
            threshold = th / 100
            recalls, precisions = calculate_metrics(preds, targets, threshold)
            recalls_with_cond, precisions_with_cond = calculate_metrics_with_conditions(
                preds, targets, threshold
            )

            metrics = {
                "threshold": threshold,
                "recalls": recalls,
                "precisions": precisions,
                "recalls_with_cond": recalls_with_cond,
                "precisions_with_cond": precisions_with_cond,
            }
            metrics_list.append(metrics)

            # 最初の閾値でのみPR-AUCを計算
            if th == 5:
                for name in recalls.keys():
                    pr_aucs[name] = calculate_prauc(
                        [m["recalls"][name] for m in metrics_list],
                        [m["precisions"][name] for m in metrics_list],
                    )

        mean_pr_auc = sum(pr_aucs.values()) / len(pr_aucs)
        return eval_loss / len(eval_loader), mean_pr_auc, metrics_list

    def save_model(self, path: str) -> None:
        """モデルを保存する.

        Args:
            path: 保存先パス
        """
        torch.save(self.model.state_dict(), path)

    def save_loss_log(self) -> None:
        """損失のログを保存する."""
        log_path = os.path.join(self.config.save_dir, "loss_log.txt")
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,eval_loss\n")
            for i, (tl, el) in enumerate(zip(self.state.train_losses, self.state.eval_losses)):
                f.write(f"{i},{tl:.4f},{el:.4f}\n")

    def train(
        self, train_loader: DataLoader, eval_loader: DataLoader, num_epochs: int
    ) -> TrainState:
        """学習を実行する.

        Args:
            train_loader: 学習用データローダー
            eval_loader: 評価用データローダー
            num_epochs: エポック数

        Returns:
            学習の状態
        """
        for epoch in range(num_epochs):
            # 学習
            train_loss = self.train_epoch(epoch, train_loader, self.config.smooth_alpha)
            self.state.train_losses.append(train_loss)

            # 評価
            eval_loss, mean_pr_auc, metrics = self.evaluate(eval_loader)
            self.state.eval_losses.append(eval_loss)

            # 学習率の更新
            if self.scheduler is not None:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # モデルの保存
            if eval_loss < self.state.best_eval_loss:
                self.state.best_eval_loss = eval_loss
                self.save_model(os.path.join(self.config.save_dir, "best_loss_model.pth"))

            if mean_pr_auc > self.state.best_mean_pr_auc:
                self.state.best_mean_pr_auc = mean_pr_auc
                self.save_model(os.path.join(self.config.save_dir, "best_prauc_model.pth"))

            # 進捗の表示
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"LR: {current_lr:.6f}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Eval Loss: {eval_loss:.4f}, "
                f"PR-AUC: {mean_pr_auc:.4f}"
            )

        # 学習ログの保存
        self.save_loss_log()

        return self.state
