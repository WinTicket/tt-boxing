"""マルチフレーム学習実行スクリプト."""
from __future__ import annotations

import argparse

import torch
from torch.optim.lr_scheduler import StepLR

from config.train_config import TrainConfig
from core.scheduler import CosineAnnealingLR
from core.trainer import Trainer
from utils.train_utils import (
    seed_everything,
    setup_criterion_and_optimizer,
    setup_datasets,
    setup_model,
    setup_transforms,
)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする.

    Returns:
        パース済みの引数
    """
    parser = argparse.ArgumentParser(description="マルチフレーム学習スクリプト")
    parser.add_argument("--config", type=str, help="設定ファイルのパス")
    return parser.parse_args()


def main() -> None:
    """メイン処理."""
    # 設定の読み込み
    config = TrainConfig()
    config.validate()

    # シードの固定
    seed_everything(42)

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの準備
    model = setup_model(config)
    model.to(device)

    # データ変換の準備
    train_transform, eval_transform = setup_transforms(config)

    # データセットとデータローダーの準備
    train_loader, eval_loader = setup_datasets(config, train_transform, eval_transform)

    # 損失関数とオプティマイザの準備
    criterion, optimizer, weights_tensor = setup_criterion_and_optimizer(config, model, device)

    # スケジューラの準備
    scheduler = None
    if config.scheduler_type == "cosinewarmup":
        scheduler = CosineAnnealingLR(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            max_epochs=config.num_epochs - 5,
        )
    elif config.scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=config.num_epochs - 5, gamma=0.1)

    # トレーナーの準備
    trainer = Trainer(
        config=config,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        weights_tensor=weights_tensor,
        device=device,
    )

    # 学習の実行
    trainer.train(train_loader, eval_loader, config.num_epochs)


if __name__ == "__main__":
    main()
