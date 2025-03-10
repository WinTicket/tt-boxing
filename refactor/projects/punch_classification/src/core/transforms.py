"""画像変換のためのモジュール."""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch


class Compose:
    """複数の変換を組み合わせるクラス."""

    def __init__(self, transforms: list) -> None:
        """初期化.

        Args:
            transforms: 変換関数のリスト
        """
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[np.ndarray | list, np.ndarray | list]:
        """変換を適用.

        Args:
            image: 入力画像
            label: 入力ラベル

        Returns:
            変換後の画像とラベル
        """
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label


class BaseAugmentation:
    """データ拡張の基底クラス."""

    def __init__(self, p: float = 0.5, is_video_clip: bool = False) -> None:
        """初期化.

        Args:
            p: 適用確率
            is_video_clip: ビデオクリップモードかどうか
        """
        self.p = p
        self.is_video_clip = is_video_clip

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[np.ndarray | list, np.ndarray | list]:
        """変換を適用."""
        pass


class ToTensor(BaseAugmentation):
    """NumPy配列をTensorに変換するクラス."""

    def __init__(self, p: float = 1.0, is_video_clip: bool = False) -> None:
        """初期化."""
        super().__init__(p, is_video_clip)

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """変換を適用.

        Args:
            image: 入力画像
            label: 入力ラベル

        Returns:
            変換後の画像とラベル
        """
        if self.is_video_clip:
            image = np.concatenate([img[np.newaxis, :, :, :] for img in image])
            label = np.concatenate([lb[np.newaxis, :] for lb in label])
            return torch.tensor(image).float(), torch.tensor(label).float()

        return torch.tensor(image).float(), torch.tensor(label).float()


class Normalize(BaseAugmentation):
    """画像を正規化するクラス."""

    def __init__(
        self, mean: list[float], std: list[float], p: float = 1.0, is_video_clip: bool = False
    ) -> None:
        """初期化.

        Args:
            mean: 平均値
            std: 標準偏差
            p: 適用確率
            is_video_clip: ビデオクリップモードかどうか
        """
        super().__init__(p, is_video_clip)
        self.mean = np.array(mean)
        self.std = np.array(std)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """画像を正規化.

        Args:
            image: 入力画像

        Returns:
            正規化された画像
        """
        image = image.astype(float)
        image = image.transpose((2, 0, 1))  # H, W, C -> C, H, W
        image /= 255.0
        image -= self.mean.reshape(3, 1, 1)
        image /= self.std.reshape(3, 1, 1)
        return image

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[np.ndarray | list, np.ndarray | list]:
        """変換を適用.

        Args:
            image: 入力画像
            label: 入力ラベル

        Returns:
            正規化された画像とラベル
        """
        if self.is_video_clip:
            return [self.normalize(img) for img in image], label

        return self.normalize(image), label


class Resize(BaseAugmentation):
    """画像をリサイズするクラス."""

    def __init__(
        self,
        size: tuple[int, int] | int,
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1.0,
        is_video_clip: bool = False,
    ) -> None:
        """初期化.

        Args:
            size: リサイズ後のサイズ（タプルの場合は(height, width)、整数の場合は短辺をそのサイズに）
            interpolation: 補間方法
            p: 適用確率
            is_video_clip: ビデオクリップモードかどうか
        """
        super().__init__(p, is_video_clip)
        self.size = size if isinstance(size, tuple) else (size, size)
        self.interpolation = interpolation
        self.keep_aspect_ratio = not isinstance(size, tuple)

    def compute_resize_size(self, height: int, width: int) -> tuple[int, int]:
        """アスペクト比を保持する場合のリサイズ後のサイズを計算.

        Args:
            height: 元の高さ
            width: 元の幅

        Returns:
            リサイズ後のサイズ
        """
        if not self.keep_aspect_ratio:
            return self.size

        target_size = self.size[0]  # 短辺のターゲットサイズ
        aspect_ratio = width / height

        if height > width:
            new_height = target_size
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = target_size
            new_height = int(new_width / aspect_ratio)

        return (new_height, new_width)

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[np.ndarray | list, np.ndarray | list]:
        """変換を適用.

        Args:
            image: 入力画像 (H, W, C)
            label: 入力ラベル

        Returns:
            リサイズされた画像とラベル
        """
        if random.random() > self.p:
            return image, label

        if self.is_video_clip:
            height, width = image[0].shape[:2]
        else:
            height, width = image.shape[:2]

        if self.keep_aspect_ratio:
            resize_size = self.compute_resize_size(height, width)
        else:
            resize_size = (self.size[1], self.size[0])  # cv2.resizeはwidth, heightの順

        if self.is_video_clip:
            resized = [cv2.resize(img, resize_size, interpolation=self.interpolation) for img in image]
        else:
            resized = cv2.resize(image, resize_size, interpolation=self.interpolation)

        return resized, label


class HorizontalFlipAugmentationNew(BaseAugmentation):
    """画像を左右反転するデータ拡張クラス（新フォーマット対応）."""

    def __init__(self, p: float = 0.5, is_video_clip: bool = False) -> None:
        """初期化.

        Args:
            p: 反転を適用する確率
            is_video_clip: ビデオクリップモードかどうか
        """
        super().__init__(p, is_video_clip)

    def label_flip(self, label: np.ndarray) -> np.ndarray:
        """ラベルを左右反転.

        Args:
            label: 入力ラベル

        Returns:
            反転されたラベル
        """
        if label[2] == 1:  # flip arm label left -> right
            label[2] = 0
            label[3] = 1
        elif label[3] == 1:  # flip arm label right -> left
            label[3] = 0
            label[2] = 1

        if label[22] == 1:  # flip hit location label temple-left -> temple-right
            label[22] = 0
            label[23] = 1
        elif label[23] == 1:  # flip hit location label temple-right -> temple-left
            label[23] = 0
            label[22] = 1

        if label[24] == 1:  # flip hit location label jaw-left -> jaw-right
            label[24] = 0
            label[25] = 1
        elif label[25] == 1:  # flip hit location label jaw-right -> jaw-left
            label[25] = 0
            label[24] = 1

        if label[36] == 1:  # flip hit location label globe-left -> globe-right
            label[36] = 0
            label[37] = 1
        elif label[37] == 1:  # flip hit location label globe-right -> globe-left
            label[37] = 0
            label[36] = 1

        if label[38] == 1:  # flip hit location label arm-left -> arm-right
            label[38] = 0
            label[39] = 1
        elif label[39] == 1:  # flip hit location label arm-right -> arm-left
            label[39] = 0
            label[38] = 1
        return label

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[np.ndarray | list, np.ndarray | list]:
        """変換を適用.

        Args:
            image: 入力画像 (H, W, C)
            label: 入力ラベル

        Returns:
            反転された画像とラベル
        """
        if random.random() < self.p:
            if self.is_video_clip:
                return [np.fliplr(img).copy() for img in image], [self.label_flip(lb) for lb in label]

            return np.fliplr(image).copy(), self.label_flip(label)

        return image, label


class HorizontalFlipAugmentation(BaseAugmentation):
    """画像を左右反転するデータ拡張クラス."""

    def __init__(self, p: float = 0.5, is_video_clip: bool = False) -> None:
        """初期化.

        Args:
            p: 反転を適用する確率
            is_video_clip: ビデオクリップモードかどうか
        """
        super().__init__(p, is_video_clip)

    def label_flip(self, label: np.ndarray) -> np.ndarray:
        """ラベルを左右反転.

        Args:
            label: 入力ラベル

        Returns:
            反転されたラベル
        """
        if label[2] == 1:  # flip arm label left -> right
            label[2] = 0
            label[3] = 1
        elif label[3] == 1:  # flip arm label right -> left
            label[3] = 0
            label[2] = 1

        if label[23] == 1:  # flip hit location label head-left -> head-right
            label[23] = 0
            label[24] = 1
        elif label[24] == 1:  # flip hit location label head-right -> head-left
            label[24] = 0
            label[23] = 1

        if label[26] == 1:  # flip hit location label body-left -> body-right
            label[26] = 0
            label[27] = 1
        elif label[27] == 1:  # flip hit location label body-right -> body-left
            label[27] = 0
            label[26] = 1

        if label[29] == 1:  # flip hit location label globe-left -> globe-right
            label[29] = 0
            label[30] = 1
        elif label[30] == 1:  # flip hit location label globe-right -> globe-left
            label[30] = 0
            label[29] = 1

        if label[31] == 1:  # flip hit location label arm-left -> arm-right
            label[31] = 0
            label[32] = 1
        elif label[32] == 1:  # flip hit location label arm-right -> arm-left
            label[32] = 0
            label[31] = 1
        return label

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[np.ndarray | list, np.ndarray | list]:
        """変換を適用.

        Args:
            image: 入力画像 (H, W, C)
            label: 入力ラベル

        Returns:
            反転された画像とラベル
        """
        if random.random() < self.p:
            if self.is_video_clip:
                return [np.fliplr(img).copy() for img in image], [self.label_flip(lb) for lb in label]

            return np.fliplr(image).copy(), self.label_flip(label)

        return image, label


class RandomCropAugmentation(BaseAugmentation):
    """画像をランダムにクロップするデータ拡張クラス."""

    def __init__(self, crop_size: tuple[int, int], p: float = 0.5, is_video_clip: bool = False) -> None:
        """初期化.

        Args:
            crop_size: 出力画像サイズ (height, width)
            p: データ拡張を適用する確率
            is_video_clip: ビデオクリップモードかどうか
        """
        super().__init__(p, is_video_clip)
        self.crop_height, self.crop_width = crop_size

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[np.ndarray | list, np.ndarray | list]:
        """変換を適用.

        Args:
            image: 入力画像 (H, W, C)
            label: 入力ラベル

        Returns:
            クロップされた画像とラベル
        """
        if random.random() > self.p:
            return image, label

        height, width = image[0].shape[:2] if self.is_video_clip else image.shape[:2]

        max_x = width - self.crop_width
        max_y = height - self.crop_height

        if max_x < 0 or max_y < 0:
            pad_width = max(0, self.crop_width - width)
            pad_height = max(0, self.crop_height - height)

            pad_width_left = pad_width // 2
            pad_width_right = pad_width - pad_width_left
            pad_height_top = pad_height // 2
            pad_height_bottom = pad_height - pad_height_top

            image = np.pad(
                image,
                ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right), (0, 0)),
                mode="constant",
            )

            height, width = image.shape[:2]
            max_x = width - self.crop_width
            max_y = height - self.crop_height

        x = random.randint(0, max(0, max_x))
        y = random.randint(0, max(0, max_y))

        if self.is_video_clip:
            cropped = [img[y:y + self.crop_height, x:x + self.crop_width] for img in image]
        else:
            cropped = image[y:y + self.crop_height, x:x + self.crop_width]

        return cropped, label


class BrightnessContrastAugmentation(BaseAugmentation):
    """画像の明るさとコントラストをランダムに変更するデータ拡張クラス."""

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
        is_video_clip: bool = False,
    ) -> None:
        """初期化.

        Args:
            brightness_range: 明るさの変更範囲 (min, max)
            contrast_range: コントラストの変更範囲 (min, max)
            p: 処理を適用する確率
            is_video_clip: ビデオクリップモードかどうか
        """
        super().__init__(p, is_video_clip)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def adjust_brightness(self, image: np.ndarray | list, factor: float) -> np.ndarray:
        """明るさを調整.

        Args:
            image: 入力画像
            factor: 明るさの調整係数

        Returns:
            明るさが調整された画像
        """
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    def adjust_contrast(self, image: np.ndarray | list, factor: float) -> np.ndarray:
        """コントラストを調整.

        Args:
            image: 入力画像
            factor: コントラストの調整係数

        Returns:
            コントラストが調整された画像
        """
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        adjusted = np.clip(mean + factor * (image - mean), 0, 255)
        return adjusted.astype(np.uint8)

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[np.ndarray | list, np.ndarray | list]:
        """変換を適用.

        Args:
            image: 入力画像 (H, W, C), uint8
            label: 入力ラベル

        Returns:
            変換後の画像とラベル
        """
        if random.random() > self.p:
            return image, label

        if self.is_video_clip:
            image = [img.astype(np.float32) for img in image]
        else:
            image = image.astype(np.float32)

        brightness_factor = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)

        if self.is_video_clip:
            image = [self.adjust_brightness(img, brightness_factor) for img in image]
            image = [self.adjust_contrast(img, contrast_factor) for img in image]
        else:
            image = self.adjust_brightness(image, brightness_factor)
            image = self.adjust_contrast(image, contrast_factor)

        return image, label


class CutoutAugmentation(BaseAugmentation):
    """画像の一部をランダムに切り取って黒く塗りつぶすデータ拡張クラス."""

    def __init__(
        self,
        n_holes: int = 1,
        length_range: tuple[int, int] = (40, 60),
        fill_value: int | tuple[int, ...] = 0,
        p: float = 0.5,
        is_video_clip: bool = False,
    ) -> None:
        """初期化.

        Args:
            n_holes: 切り取る領域の数
            length_range: 切り取る領域のサイズ範囲 (min, max)
            fill_value: 塗りつぶす値（チャンネルごとに指定可能）
            p: 処理を適用する確率
            is_video_clip: ビデオクリップモードかどうか
        """
        super().__init__(p, is_video_clip)
        self.n_holes = n_holes
        self.length_range = length_range
        self.fill_value = fill_value

    def create_mask(
        self, height: int, width: int, center_y: int, center_x: int, hole_height: int, hole_width: int
    ) -> np.ndarray:
        """切り取り領域のマスクを作成.

        Args:
            height: 画像の高さ
            width: 画像の幅
            center_y: 切り取り領域の中心Y座標
            center_x: 切り取り領域の中心X座標
            hole_height: 切り取り領域の高さ
            hole_width: 切り取り領域の幅

        Returns:
            マスク画像
        """
        y1 = np.clip(center_y - hole_height // 2, 0, height)
        y2 = np.clip(center_y + hole_height // 2, 0, height)
        x1 = np.clip(center_x - hole_width // 2, 0, width)
        x2 = np.clip(center_x + hole_width // 2, 0, width)

        mask = np.ones((height, width), np.float32)
        mask[y1:y2, x1:x2] = 0
        return mask

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[np.ndarray | list, np.ndarray | list]:
        """変換を適用.

        Args:
            image: 入力画像 (H, W, C)
            label: 入力ラベル

        Returns:
            Cutout適用後の画像とラベル
        """
        if random.random() > self.p:
            return image, label

        height, width = image[0].shape[:2] if self.is_video_clip else image.shape[:2]
        image = image.copy()

        for _ in range(self.n_holes):
            hole_height = random.randint(*self.length_range)
            hole_width = random.randint(*self.length_range)

            center_y = random.randint(0, height)
            center_x = random.randint(0, width)

            mask = self.create_mask(height, width, center_y, center_x, hole_height, hole_width)

            channel = image[0].shape[-1] if self.is_video_clip else image.shape[-1]
            mask = np.expand_dims(mask, axis=-1)
            mask = np.tile(mask, [1, 1, channel])

            if isinstance(self.fill_value, int | float):
                fill_value = [self.fill_value] * channel
            else:
                fill_value = self.fill_value

            if self.is_video_clip:
                image = [img * mask + np.array(fill_value) * (1 - mask) for img in image]
                image = [img.astype(np.uint8) for img in image]
            else:
                image = image * mask + np.array(fill_value) * (1 - mask)
                image = image.astype(np.uint8)

        return image, label


class GrayscaleAugmentation(BaseAugmentation):
    """画像をグレースケール化するデータ拡張クラス."""

    def __init__(self, p: float = 0.5, is_video_clip: bool = False) -> None:
        """初期化."""
        super().__init__(p, is_video_clip)

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[np.ndarray | list, np.ndarray | list]:
        """変換を適用.

        Args:
            image: 入力画像 (H, W, C) RGB format
            label: 入力ラベル

        Returns:
            グレースケール化された画像とラベル
        """
        if np.random.random() > self.p:
            return image, label

        if self.is_video_clip:
            gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in image]
            gray_3ch = [cv2.cvtColor(gr, cv2.COLOR_GRAY2BGR) for gr in gray]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return gray_3ch, label


class HSVAugmentation(BaseAugmentation):
    """HSV色空間でのデータ拡張クラス."""

    def __init__(
        self,
        p: float = 0.5,
        is_video_clip: bool = False,
        hue_shift_limit: tuple[int, int] = (-20, 20),
        sat_shift_limit: tuple[int, int] = (-30, 30),
        val_shift_limit: tuple[int, int] = (-30, 30),
    ) -> None:
        """初期化.

        Args:
            p: 適用確率
            is_video_clip: ビデオクリップモードかどうか
            hue_shift_limit: 色相のシフト範囲
            sat_shift_limit: 彩度のシフト範囲
            val_shift_limit: 明度のシフト範囲
        """
        super().__init__(p, is_video_clip)
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit

    def apply_hsv_transform(self, image: np.ndarray) -> np.ndarray:
        """HSV変換を適用.

        Args:
            image: 入力画像

        Returns:
            HSV変換後の画像
        """
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        hue_shift = np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
        sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
        val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])

        image_hsv[..., 0] = np.clip(image_hsv[..., 0] + hue_shift, 0, 179)  # Hue
        image_hsv[..., 1] = np.clip(image_hsv[..., 1] + sat_shift, 0, 255)  # Saturation
        image_hsv[..., 2] = np.clip(image_hsv[..., 2] + val_shift, 0, 255)  # Value

        return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    def __call__(
        self, image: np.ndarray | list, label: np.ndarray | list
    ) -> tuple[np.ndarray | list, np.ndarray | list]:
        """変換を適用.

        Args:
            image: 入力画像
            label: 入力ラベル

        Returns:
            HSV変換後の画像とラベル

        Raises:
            ValueError: 入力形式が不正な場合
        """
        if np.random.random() > self.p:
            return image, label

        if self.is_video_clip:
            if not isinstance(image, list) or not isinstance(label, list):
                raise ValueError("Video clip mode requires list inputs")

            augmented_images = []
            for img in image:
                augmented_images.append(self.apply_hsv_transform(img))
            return augmented_images, label

        if isinstance(image, list) or isinstance(label, list):
            raise ValueError("Single image mode requires numpy array inputs")

        return self.apply_hsv_transform(image), label
