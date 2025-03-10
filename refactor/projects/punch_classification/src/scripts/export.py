"""各フレームの検知結果をJSONに出力するモジュール.

対応モデル：2.5D DINOv2, 2.5D ResNet
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from core.dataset import WinTicketMultiFrameDataset
from utils.train_utils import setup_datasets, setup_model


def process_dataset(
    dataset: WinTicketMultiFrameDataset, model: nn.Module, viewpoint: str
) -> dict[str, list[float]]:
    """データセットを処理し、検出結果を取得する.

    Args:
        dataset: 評価用データセット
        model: 評価用モデル
        viewpoint: 視点 ("rb", "rt", "all")

    Returns:
        ファイル名と検出結果のマッピング
    """
    result_dict = {}

    with tqdm(range(len(dataset))) as t:
        for i in t:
            filename = dataset.filenames[i]
            if filename.endswith(f"{viewpoint}_0.jpg") or filename.endswith(f"{viewpoint}_1.jpg"):
                input_data, _ = dataset[i]

                torch.cuda.synchronize()
                get_data_end = time.time()

                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(input_data.unsqueeze(0).to("cuda"))

                torch.cuda.synchronize()
                end = time.time()

                t.postfix = str(end - get_data_end)
                t.update()

                result_dict[filename] = output.cpu().to(torch.float32).numpy()[0].tolist()

    return result_dict


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(description="各フレームの検知結果をJSONに出力するスクリプト")
    parser.add_argument("-a", "--annotation_path", required=True, help="アノテーションファイルパス")
    parser.add_argument("-c", "--checkpoint_path", required=True, help="チェックポイントファイルパス")
    parser.add_argument("-sd", "--savedir", required=True, help="保存ディレクトリ")
    parser.add_argument("-op", "--output_filename", help="出力ファイル名")
    parser.add_argument(
        "-v", "--viewpoint", choices=["rb", "rt", "all"], default="rb", help="視点の選択"
    )
    args = parser.parse_args()

    # 設定
    root_dir = "data/winticket_boxing/crop_frames/"
    input_size = (224, 224)
    model_type = "resnet50"

    # モデルの設定
    model = setup_model(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # チェックポイントの読み込み
    ckpt = torch.load(args.checkpoint_path)
    model.load_state_dict(ckpt)
    model.eval()
    model = torch.compile(model)

    # データセットの設定
    eval_dataset = setup_datasets(root_dir, args.annotation_path, input_size)

    # 検出結果の取得
    result_dict = process_dataset(eval_dataset, model, args.viewpoint)

    # 結果の保存
    os.makedirs(args.savedir, exist_ok=True)
    with open(os.path.join(args.savedir, args.output_filename), "w") as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    main()
