"""パンチ分類のためのデータセットモジュール."""
from __future__ import annotations

import json
import os
import random
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class WinticketBoxingPunchClassficationDataset(Dataset):
    """Winticketボクシングパンチ分類用のデータセット."""

    def __init__(
        self,
        root_dir: str,
        train_datalist: str,
        val_datalist: str,
        transform: callable | None = None,
        is_train: bool = True,
        num_per_classes: list[int] = [1, 1, 2, 6, 2, 4, 3, 3, 13, 3],
        original_transform: bool = False,
    ) -> None:
        """初期化.

        Args:
            root_dir: データセットのルートディレクトリ
            train_datalist: 学習データリストのJSONファイルパス
            val_datalist: 検証データリストのJSONファイルパス
            transform: 変換関数
            is_train: 学習モードかどうか
            num_per_classes: クラスごとのラベル数
            original_transform: オリジナルの変換を使用するかどうか
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.num_per_classes = num_per_classes
        self.num_classes = sum(num_per_classes)
        self.original_transform = original_transform

        if self.is_train:
            assert train_datalist.endswith(".json")
            self.filenames, self.labels = self._get_datalist(train_datalist)
        else:
            assert val_datalist.endswith(".json")
            self.filenames, self.labels = self._get_datalist(val_datalist)

    def _get_datalist(self, datalist: str) -> tuple[list[str], list[np.ndarray]]:
        """データリストを取得.

        Args:
            datalist: データリストのJSONファイルパス

        Returns:
            ファイル名とラベルのタプル
        """
        with open(datalist) as f:
            json_annotaions = json.load(f)

        filenames = list(json_annotaions.keys())
        labels = list(json_annotaions.values())
        mutltihot_labels = []
        for label in labels:
            multihot_label = []
            for nc, lb in zip(self.num_per_classes, label):
                zero_vector = np.zeros(nc)
                if lb != 0:
                    zero_vector[lb - 1] = 1
                multihot_label.append(zero_vector)
            multihot_label = np.concatenate(multihot_label)
            mutltihot_labels.append(multihot_label)

        return filenames, mutltihot_labels

    def __len__(self) -> int:
        """データセットの長さを返す."""
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[Image.Image | torch.Tensor, torch.Tensor]:
        """指定されたインデックスのアイテムを取得."""
        return self.pull_item(index)

    def pull_item(self, index: int) -> tuple[Image.Image | torch.Tensor, torch.Tensor]:
        """指定されたインデックスのアイテムを取得.

        Args:
            index: データのインデックス

        Returns:
            画像とラベルのタプル
        """
        filename = self.filenames[index]
        labels = self.labels[index]

        imagepath = os.path.join(self.root_dir, filename)
        image = Image.open(imagepath).convert("RGB")

        if self.transform is not None:
            if self.original_transform:
                image = np.array(image)
                labels = np.array(labels)
                image, labels = self.transform(image, labels)
                return image, labels

            image = self.transform(image)
            return image, torch.tensor(labels).float()


class WinTicketMultiFrameDataset(WinticketBoxingPunchClassficationDataset):
    """3フレームをチャネル方向に結合するデータセット.

    t-2, t-1, tのフレームがそれぞれRGBに割り当てられる
    """

    def __init__(
        self,
        root_dir: str,
        train_datalist: str,
        val_datalist: str,
        transform: callable | None = None,
        is_train: bool = True,
        num_per_classes: list[int] = [1, 1, 2, 6, 2, 4, 3, 3, 13, 3],
        frame_span: int = 2,
        size: tuple[int, int] = (256, 256),
        enable_mixup: bool = False,
        mixup_prob: float = 0.5,
        mixup_alpha: float = 0.5,
    ) -> None:
        """初期化.

        Args:
            root_dir: データセットのルートディレクトリ
            train_datalist: 学習データリストのJSONファイルパス
            val_datalist: 検証データリストのJSONファイルパス
            transform: 変換関数
            is_train: 学習モードかどうか
            num_per_classes: クラスごとのラベル数
            frame_span: フレーム間の間隔
            size: 画像サイズ
            enable_mixup: Mixupを有効にするかどうか
            mixup_prob: Mixupの確率
            mixup_alpha: Mixupのアルファ値
        """
        super().__init__(root_dir, train_datalist, val_datalist, transform, is_train, num_per_classes)
        self.frame_span = frame_span
        self.size = size
        self.is_train = is_train
        self.enable_mixup = enable_mixup
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

    def _load_frame_and_label(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """フレームとラベルを読み込む.

        Args:
            index: データのインデックス

        Returns:
            フレームとラベルのタプル
        """
        filename = self.filenames[index]
        label = self.labels[index]

        image_path = os.path.join(self.root_dir, filename)

        image = Image.open(image_path).convert("L")
        image_arry = np.array(image)
        image_arry = cv2.resize(image_arry, dsize=(self.size[0], self.size[1]))
        image_arry = image_arry.reshape(image_arry.shape[0], image_arry.shape[1], 1)
        frame_stack = [image_arry]

        frame_no = int(filename.split("_")[-3])
        for _ in [1, 2]:
            if self.is_train:
                span = random.choice(np.arange(1, self.frame_span + 1, 1))
            else:
                span = self.frame_span
            frame_no = frame_no - span

            frame_no_pattern = r"_(\d{6})_"
            bf_filename = re.sub(frame_no_pattern, f"_{str(frame_no).zfill(6)}_", filename)

            bf_image_path = os.path.join(self.root_dir, bf_filename)
            if os.path.exists(bf_image_path):
                bf_image = Image.open(bf_image_path).convert("L")
                bf_image_arry = np.array(bf_image)
            else:
                bf_image_arry = np.zeros_like(image_arry, dtype=image_arry.dtype)

            bf_image_arry = cv2.resize(bf_image_arry, dsize=(self.size[0], self.size[1]))
            bf_image_arry = bf_image_arry.reshape(image_arry.shape[0], image_arry.shape[1], 1)
            frame_stack.append(bf_image_arry)

        frame_stack = np.concatenate(frame_stack, axis=-1)
        return frame_stack, label

    def pull_item(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """指定されたインデックスのアイテムを取得.

        Args:
            index: データのインデックス

        Returns:
            フレームスタックとラベルのタプル
        """
        frame_stack, label = self._load_frame_and_label(index)

        if self.transform is not None:
            frame_stack, label = self.transform(frame_stack, label)

        if self.is_train and self.enable_mixup and random.random() < self.mixup_prob:
            mixup_index = random.randint(0, len(self) - 1)
            frame_stack2, label2 = self._load_frame_and_label(mixup_index)
            if self.transform is not None:
                frame_stack2, label2 = self.transform(frame_stack2, label2)

            alpha = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            frame_stack = alpha * frame_stack + (1 - alpha) * frame_stack2
            label = alpha * label + (1 - alpha) * label2

        return frame_stack, label


class WinTicketVideoClipDataset(WinticketBoxingPunchClassficationDataset):
    """ビデオクリップデータセット."""

    def __init__(
        self,
        root_dir: str,
        train_datalist: str,
        val_datalist: str,
        len_clip: int,
        frame_span: int,
        size: tuple[int, int] = (256, 256),
        transform: callable | None = None,
        is_train: bool = True,
        num_per_classes: list[int] = [1, 1, 2, 6, 2, 4, 3, 3, 13, 3],
        original_transform: bool = False,
    ) -> None:
        """初期化.

        Args:
            root_dir: データセットのルートディレクトリ
            train_datalist: 学習データリストのJSONファイルパス
            val_datalist: 検証データリストのJSONファイルパス
            len_clip: クリップの長さ
            frame_span: フレーム間の間隔
            size: 画像サイズ
            transform: 変換関数
            is_train: 学習モードかどうか
            num_per_classes: クラスごとのラベル数
            original_transform: オリジナルの変換を使用するかどうか
        """
        super().__init__(
            root_dir,
            train_datalist,
            val_datalist,
            transform,
            is_train,
            num_per_classes,
            original_transform,
        )
        self.len_clip = len_clip
        self.frame_span = frame_span
        self.size = size
        self.filename2label = {fn: lb for fn, lb in zip(self.filenames, self.labels, strict=True)}

    def pull_item(self, index: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """指定されたインデックスのアイテムを取得.

        Args:
            index: データのインデックス

        Returns:
            ビデオクリップとラベルのタプル
        """
        video_clips = []
        labels = []

        filename = self.filenames[index]
        label = self.labels[index]

        image_path = os.path.join(self.root_dir, filename)

        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.size)
        image_arry = np.array(image)

        video_clips.append(image_arry)
        labels.append(label)

        for i in range(1, self.len_clip):
            frame_no = int(filename.split("_")[-3])
            bf_frame_no = frame_no - i * self.frame_span

            frame_no_pattern = r"_(\d{6})_"
            bf_filename = re.sub(frame_no_pattern, f"_{str(bf_frame_no).zfill(6)}_", filename)

            bf_image_path = os.path.join(self.root_dir, bf_filename)
            if os.path.exists(bf_image_path):
                bf_image = Image.open(bf_image_path).convert("RGB")
                bf_image = bf_image.resize(self.size)
                bf_image_arry = np.array(bf_image)
            else:
                bf_image_arry = np.zeros_like(image_arry, dtype=image_arry.dtype)

            video_clips.append(bf_image_arry)

            if bf_filename in self.filename2label:
                labels.append(self.filename2label[bf_filename])
            else:
                bf_label = np.zeros(self.num_classes)
                bf_label[0] = 1
                labels.append(bf_label)

        if self.transform is not None:
            video_clips, labels = self.transform(video_clips, labels)

        return video_clips, labels
