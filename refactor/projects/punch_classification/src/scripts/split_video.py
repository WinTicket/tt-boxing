"""動画を分割して画像として保存するモジュール."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def split_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """フレームを4つの視点に分割する.

    Args:
        frame: 入力フレーム

    Returns:
        左上、右上、左下、右下のフレームのタプル

    Raises:
        AssertionError: 分割されたフレームのサイズが一致しない場合
    """
    height, width = frame.shape[:2]
    half_h, half_w = height // 2, width // 2

    left_top_frame = frame[:half_h, :half_w]
    right_top_frame = frame[:half_h, half_w:]
    left_bottom_frame = frame[half_h:, :half_w]
    right_bottom_frame = frame[half_h:, half_w:]

    # サイズの一致を確認
    assert all(
        a == b
        for a, b in zip(left_top_frame.shape, right_top_frame.shape, strict=True)
    ), "右上のフレームサイズが一致しません"
    assert all(
        a == b
        for a, b in zip(left_top_frame.shape, right_bottom_frame.shape, strict=True)
    ), "右下のフレームサイズが一致しません"
    assert all(
        a == b
        for a, b in zip(left_top_frame.shape, left_bottom_frame.shape, strict=True)
    ), "左下のフレームサイズが一致しません"

    return left_top_frame, right_top_frame, left_bottom_frame, right_bottom_frame


def process_video(
    video_path: Path,
    save_dir: Path,
    no_split_view: bool = False,
    max_frame_cnt: int = -1,
) -> None:
    """動画を処理して画像として保存する.

    Args:
        video_path: 入力動画のパス
        save_dir: 保存先ディレクトリ
        no_split_view: 分割せずに保存するかどうか
        max_frame_cnt: 最大フレーム数（-1の場合は全フレーム）
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画ファイルを開けません: {video_path}")

    save_dir.mkdir(exist_ok=True)

    frame_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if no_split_view:
            # 1視点映像の場合
            save_path = save_dir / f"{video_path.stem}_{frame_cnt:06}_all.jpg"
            cv2.imwrite(str(save_path), frame)
        else:
            # 4視点の映像の場合
            left_top, right_top, left_bottom, right_bottom = split_frame(frame)

            # 各視点の画像を保存
            for view_frame, suffix in [
                (left_top, "lt"),
                (right_top, "rt"),
                (left_bottom, "lb"),
                (right_bottom, "rb"),
            ]:
                save_path = save_dir / f"{video_path.stem}_{frame_cnt:06}_{suffix}.jpg"
                cv2.imwrite(str(save_path), view_frame)

        frame_cnt += 1
        if max_frame_cnt > -1 and frame_cnt > max_frame_cnt:
            break

    cap.release()


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(description="動画を分割して画像として保存するスクリプト")
    parser.add_argument("-v", "--video", type=str, required=True, help="入力動画のパス")
    parser.add_argument("-sd", "--savedir", type=str, required=True, help="保存先ディレクトリ")
    parser.add_argument(
        "--no_split_view",
        action="store_true",
        default=False,
        help="分割せずに保存する",
    )
    parser.add_argument(
        "--max_frame_cnt",
        type=int,
        default=-1,
        help="最大フレーム数（-1の場合は全フレーム）",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    save_dir = Path(args.savedir)

    try:
        process_video(video_path, save_dir, args.no_split_view, args.max_frame_cnt)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
