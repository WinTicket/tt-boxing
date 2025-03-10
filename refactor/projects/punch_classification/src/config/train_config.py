"""学習設定モジュール."""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """学習設定."""

    # データパス
    train_rootdir: str = "data/winticket_boxing/crop_frames/"
    train_datalist: str = (
        "data/winticket_boxing/annotations/"
        "crop_square_20241031_boxing_4K_m1_m4_m5_m6_m3_rt_rb_yamato3_yamato6_span30.json"
    )
    eval_rootdir: str = "data/winticket_boxing/crop_frames/"
    eval_datalist: str = (
        "data/winticket_boxing/annotations/"
        "crop_square_boxing_20240822_m5_annotations_rt_rb_new.json"
    )
    save_dir: str = (
        "model/20250220_stack_multiframe_square_m1_m4_m5_m6_m3_yamato3_yamato6_span30_resnet50_rndmspn"
    )
    pretrained_weight: str | None = None

    # 画像サイズ
    train_crop: tuple[int, int] = (256, 256)
    input_size: tuple[int, int] = (224, 224)

    # 学習パラメータ
    num_epochs: int = 20
    warmup_epochs: int = 0
    lr: float = 1e-4
    wd: float = 0
    batch_size: int = 16
    num_workers: int = 4

    # データ拡張
    brightness_contrast_prob: float = 0.5
    brightness_min: float = 0.8
    brightness_max: float = 1.2
    contrast_min: float = 0.8
    contrast_max: float = 1.2
    cutout_prob: float = 0.3

    # Mixup
    enable_mixup: bool = False
    mixup_prob: float = 0.0
    mixup_alpha: float = 0.0

    # ラベルスムージング
    smooth_alpha: float = 0.0

    # モデル設定
    model_type: str = "resnet50"  # ["resnet18", "resnet50", "resnet101", "dino"]
    optimizer_type: str = "adam"  # ["adam", "adamw"]
    scheduler_type: str = "none"  # ["none", "cosinewarmup", "step"]
    loss_type: str = "bce"  # ["bce", "cbloss", "cbloss_drw"]

    def validate(self) -> None:
        """設定の妥当性を検証する."""
        assert self.model_type in ["resnet18", "resnet50", "resnet101", "dino"]
        assert self.optimizer_type in ["adam", "adamw"]
        assert self.scheduler_type in ["none", "cosinewarmup", "step"]
        assert self.loss_type in ["bce", "cbloss", "cbloss_drw"]
        if self.loss_type == "cbloss_drw":
            assert self.scheduler_type == "step", "cbloss_drwはStepLRのみ対応"
