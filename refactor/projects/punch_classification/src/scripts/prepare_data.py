"""分類器の学習時にアスペクト比を維持するため、BBoxの長辺を一辺とする正方形をクロップするモジュール."""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def create_square_crop(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """BBoxの長辺を一辺とする正方形領域をクロップする.

    Args:
        image: 入力画像
        x1: バウンディングボックスの左上x座標
        y1: バウンディングボックスの左上y座標
        x2: バウンディングボックスの右下x座標
        y2: バウンディングボックスの右下y座標

    Returns:
        クロップされた画像と新しい座標のタプル (sx1, sy1, sx2, sy2)
    """
    img_height, img_width = image.shape[:2]
    width = x2 - x1
    height = y2 - y1

    long_side = max(width, height)
    center_x = int(x1 + width / 2)
    center_y = int(y1 + height / 2)

    sx1 = max(int(center_x - long_side / 2), 0)
    sx2 = min(int(center_x + long_side / 2), img_width)
    sy1 = max(int(center_y - long_side / 2), 0)
    sy2 = min(int(center_y + long_side / 2), img_height)

    crop_img = image[sy1:sy2 + 1, sx1:sx2 + 1]
    return crop_img, (sx1, sy1, sx2, sy2)


def process_annotations(
    annotation_path: str,
    image_dir: str,
    crop_image_root: str,
    is_read_and_write_image: bool = True,
) -> dict[str, list[float]]:
    """アノテーションを処理し、正方形クロップを作成する.

    Args:
        annotation_path: アノテーションファイルのパス
        image_dir: 元画像のディレクトリ
        crop_image_root: クロップ画像の保存先ルートディレクトリ
        is_read_and_write_image: 画像の読み書きを行うかどうか

    Returns:
        新しいアノテーション辞書

    Raises:
        FileNotFoundError: 画像ファイルが存在しない場合
    """
    with open(annotation_path) as f:
        data = json.load(f)

    new_annotation = {}

    for filename, annotation_list in data.items():
        if is_read_and_write_image:
            image_path = os.path.join(image_dir, filename)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"{image_path} is not exist")
            image = cv2.imread(image_path)

        for i, annotation in enumerate(annotation_list):
            x1, y1, x2, y2 = map(int, annotation[:4])
            label = annotation[4:]

            if is_read_and_write_image:
                crop_img, (sx1, sy1, sx2, sy2) = create_square_crop(image, x1, y1, x2, y2)

            crop_filename = filename.replace(".jpg", f"_{i}.jpg").split(os.sep)[-1]
            crop_dir = "square_" + filename.split(os.sep)[0]

            if is_read_and_write_image:
                crop_dir_path = os.path.join(crop_image_root, crop_dir)
                os.makedirs(crop_dir_path, exist_ok=True)
                cv2.imwrite(os.path.join(crop_dir_path, crop_filename), crop_img)

            new_annotation[os.path.join(crop_dir, crop_filename)] = label

    return new_annotation


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(description="BBoxの長辺を一辺とする正方形クロップを作成するスクリプト")
    parser.add_argument(
        "-ad", "--annotation_dir", default="data/winticket_boxing/annotations", help="アノテーションディレクトリ"
    )
    parser.add_argument("-an", "--annotation_name", help="アノテーションファイル名")
    args = parser.parse_args()

    # 設定
    image_dir = "data/winticket_boxing/frames/"
    crop_image_root = "data/winticket_boxing/crop_frames/"
    new_annotation_dir = "data/winticket_boxing/annotations"
    is_read_and_write_image = True

    # 出力ディレクトリの作成
    os.makedirs(crop_image_root, exist_ok=True)

    # アノテーションの処理
    annotation_path = os.path.join(args.annotation_dir, args.annotation_name)
    new_annotation = process_annotations(
        annotation_path, image_dir, crop_image_root, is_read_and_write_image
    )

    # 新しいアノテーションの保存
    new_annotation_name = "crop_square_" + args.annotation_name
    new_annotation_path = os.path.join(new_annotation_dir, new_annotation_name)
    with open(new_annotation_path, "w") as f:
        json.dump(new_annotation, f)


if __name__ == "__main__":
    main()
