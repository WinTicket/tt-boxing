"""学習用ユーティリティモジュール."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.train_config import TrainConfig
from core.dataset import WinTicketMultiFrameDataset
from core.transforms import (
    BrightnessContrastAugmentation,
    Compose,
    CutoutAugmentation,
    HorizontalFlipAugmentationNew,
    Normalize,
    RandomCropAugmentation,
    Resize,
    ToTensor,
)


def seed_everything(seed: int = 42) -> None:
    """再現性のためにシードを固定する.

    Args:
        seed: 固定するシード値
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_model(config: TrainConfig) -> nn.Module:
    """モデルを設定する.

    Args:
        config: 学習設定

    Returns:
        設定されたモデル
    """
    if config.model_type == "dino":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        model.head = nn.Sequential(
            nn.Linear(768, 43),
            nn.Sigmoid(),
        )
    elif config.model_type == "resnet18":
        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(512, 43),
            nn.Sigmoid(),
        )
    elif config.model_type == "resnet50":
        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(2048, 43),
            nn.Sigmoid(),
        )
    else:  # resnet101
        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet101", pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(2048, 43),
            nn.Sigmoid(),
        )

    if config.pretrained_weight is not None:
        ckpt = torch.load(config.pretrained_weight)
        model.load_state_dict(ckpt)

    return model


def setup_transforms(config: TrainConfig) -> tuple[Compose, Compose]:
    """データ変換を設定する.

    Args:
        config: 学習設定

    Returns:
        学習用と評価用の変換のタプル
    """
    train_transform = Compose(
        [
            RandomCropAugmentation(crop_size=config.input_size, p=1.0),
            HorizontalFlipAugmentationNew(p=0.5),
            BrightnessContrastAugmentation(
                p=config.brightness_contrast_prob,
                brightness_range=(config.brightness_min, config.brightness_max),
                contrast_range=(config.contrast_min, config.contrast_max),
            ),
            CutoutAugmentation(p=config.cutout_prob),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensor(p=1.0),
        ]
    )

    eval_transform = Compose(
        [
            Resize(size=config.input_size, p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensor(p=1.0),
        ]
    )

    return train_transform, eval_transform


def setup_datasets(
    config: TrainConfig, train_transform: Compose, eval_transform: Compose
) -> tuple[DataLoader, DataLoader]:
    """データセットとデータローダーを設定する.

    Args:
        config: 学習設定
        train_transform: 学習用データ変換
        eval_transform: 評価用データ変換

    Returns:
        学習用と評価用のデータローダーのタプル
    """
    train_dataset = WinTicketMultiFrameDataset(
        config.train_rootdir,
        config.train_datalist,
        None,
        train_transform,
        True,
        size=config.train_crop,
        num_per_classes=[1, 1, 2, 6, 2, 4, 3, 3, 18, 3],
        enable_mixup=config.enable_mixup,
        mixup_prob=config.mixup_prob,
        mixup_alpha=config.mixup_alpha,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    eval_dataset = WinTicketMultiFrameDataset(
        config.eval_rootdir,
        None,
        config.eval_datalist,
        eval_transform,
        False,
        num_per_classes=[1, 1, 2, 6, 2, 4, 3, 3, 18, 3],
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
    )

    return train_loader, eval_loader


def get_cbloss_weight(dataset: WinTicketMultiFrameDataset, beta: float = 0.999) -> np.ndarray:
    """CBLossの重みを計算する.

    Args:
        dataset: データセット
        beta: 重みの計算に使用するパラメータ

    Returns:
        各クラスの重み
    """
    dataset_copy = copy.deepcopy(dataset)
    dataset_copy.is_train = False
    dataset_copy.transform = None

    targets = []
    for i in tqdm(range(len(dataset_copy)), desc="CBLoss重みの計算中"):
        _, target = dataset_copy.pull_item(i)
        targets.append(target)

    num_per_classes = np.array(targets).sum(axis=0)
    num_per_classes[num_per_classes == 0] = np.max(num_per_classes)

    effective_num = 1.0 - np.power(beta, num_per_classes)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * 43

    return weights


def setup_criterion_and_optimizer(
    config: TrainConfig, model: nn.Module, device: torch.device
) -> tuple[nn.Module | callable, Optimizer, torch.Tensor | None]:
    """損失関数とオプティマイザを設定する.

    Args:
        config: 学習設定
        model: モデル
        device: デバイス

    Returns:
        損失関数とオプティマイザのタプル
    """
    weights_tensor = None
    if config.loss_type == "bce":
        criterion = nn.BCELoss()
    else:  # cbloss or cbloss_drw
        criterion = nn.functional.binary_cross_entropy
        weights = np.load("cbloss_weight.npy")
        weights_tensor = torch.tensor(weights).float().unsqueeze(0).to(device)

    if config.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
    else:  # adamw
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)

    return criterion, optimizer, weights_tensor
