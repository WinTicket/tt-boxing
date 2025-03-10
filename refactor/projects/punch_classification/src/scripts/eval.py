"""Precision-Recall評価を行うモジュール."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.dataset import WinTicketMultiFrameDataset
from core.transforms import Compose, Normalize, Resize, ToTensor
from utils.eval_utils import calculate_metrics, save_metrics_log
from utils.train_utils import setup_model


def get_predictions(
    model: nn.Module, eval_loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """モデルの予測を取得する.

    Args:
        model: 評価対象のモデル
        eval_loader: 評価用データローダー
        device: 使用デバイス

    Returns:
        予測値と正解ラベルのタプル
    """
    preds = []
    targets = []
    for input_data, target in tqdm(eval_loader):
        with torch.no_grad():
            output = model(input_data.to(device))
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(description="Precision-Recall評価スクリプト")
    parser.add_argument("-c", "--checkpoint_path", required=True, help="モデルのチェックポイントパス")
    parser.add_argument("-a", "--annotation_path", required=True, help="アノテーションファイルパス")
    parser.add_argument("-o", "--output_dir", required=True, help="出力ディレクトリ")
    parser.add_argument("-m", "--model", choices=["resnet50", "dino"], default="resnet50", help="モデルの種類")
    args = parser.parse_args()

    # 設定
    val_root_dir = "data/winticket_boxing/crop_frames/"
    input_size = (224, 224)
    train_crop = (256, 256)
    batch_size = 32
    thresholds = range(5, 100, 5)

    # 出力ディレクトリの作成
    output_dir = args.output_dir
    output_path = os.path.join(output_dir, "precision_recall_curve.csv")
    os.makedirs(output_dir, exist_ok=True)

    # モデルの設定
    model = setup_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # チェックポイントの読み込み
    ckpt = torch.load(args.checkpoint_path)
    model.load_state_dict(ckpt)
    model.eval()

    # データセットとデータローダーの設定
    eval_transform = Compose(
        [
            Resize(size=input_size, p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensor(p=1.0),
        ]
    )

    eval_dataset = WinTicketMultiFrameDataset(
        val_root_dir,
        None,
        args.annotation_path,
        eval_transform,
        False,
        size=train_crop,
        num_per_classes=[1, 1, 2, 6, 2, 4, 3, 3, 18, 3],
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    # 予測の取得
    print("予測値と正解ラベルの取得中...")
    preds, targets = get_predictions(model, eval_loader, device)

    # 各閾値での評価
    all_recalls = []
    all_precisions = []
    all_only_punch_recalls = []
    all_only_punch_precisions = []

    print("各閾値での評価中...")
    for th in thresholds:
        threshold = th / 100
        recalls, precisions, only_punch_recalls, only_punch_precisions = calculate_metrics(
            preds, targets, threshold
        )

        all_recalls.append(recalls)
        all_precisions.append(precisions)
        all_only_punch_recalls.append(only_punch_recalls)
        all_only_punch_precisions.append(only_punch_precisions)

        # 結果の表示
        class_names = ["punch", "hit", "hit_no", "eff_no", "eff_mild", "eff_full"]
        for name, recall, precision, only_recall, only_precision in zip(
            class_names, recalls, precisions, only_punch_recalls, only_punch_precisions
        ):
            print(f"{name} recall: {recall:.4f}")
            print(f"{name} precision: {precision:.4f}")
            print(f"{name} only punch recall: {only_recall:.4f}")
            print(f"{name} only punch precision: {only_precision:.4f}")
        print("=" * 40)

    # 結果の書き出し
    save_metrics_log(output_path, thresholds, all_recalls, all_precisions)


if __name__ == "__main__":
    main()
