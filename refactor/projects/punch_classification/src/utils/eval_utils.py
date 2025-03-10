from pathlib import Path
import numpy as np
from sklearn.metrics import auc


def calculate_prauc(recalls: list[float], precisions: list[float]) -> float:
    """PR-AUCを計算する.

    Args:
        recalls: リコールのリスト
        precisions: 適合率のリスト

    Returns:
        PR-AUC値
    """
    recalls = np.append(np.array(recalls), [0])
    precisions = np.append(np.array(precisions), [1])

    recalls = np.nan_to_num(recalls)
    precisions = np.nan_to_num(precisions)

    sorted_indices = np.argsort(recalls)[::-1]
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    return auc(recalls, precisions)


def calculate_metrics(
    preds: np.ndarray, targets: np.ndarray, threshold: float
) -> tuple[dict[str, float], dict[str, float]]:
    """各クラスの評価指標を計算する.

    Args:
        preds: 予測値
        targets: 正解ラベル
        threshold: 判定閾値

    Returns:
        リコールと適合率の辞書のタプル
    """
    class_indices = {
        "punch": 1,
        "hit": 17,
        "hit_no": 16,
        "eff_no": 40,
        "eff_mild": 41,
        "eff_full": 42,
    }

    recalls = {}
    precisions = {}

    for name, idx in class_indices.items():
        target_count = sum(targets[:, idx] == 1)
        pred_count = sum(preds[:, idx] >= threshold)
        correct_count = sum((preds[:, idx] >= threshold) & (targets[:, idx] == 1))

        recall = correct_count / target_count
        precision = correct_count / (pred_count + 1e-12)

        recalls[name] = recall
        precisions[name] = precision

    return recalls, precisions


def calculate_metrics_with_conditions(
    preds: np.ndarray, targets: np.ndarray, threshold: float
) -> tuple[dict[str, float], dict[str, float]]:
    """条件付きの評価指標を計算する.

    Args:
        preds: 予測値
        targets: 正解ラベル
        threshold: 判定閾値

    Returns:
        リコールと適合率の辞書のタプル
    """
    punch_index = preds[:, 1] >= threshold
    hit_index = preds[:, 17] >= threshold
    hit_larger_index = preds[:, 17] >= preds[:, 16]

    recalls = {}
    precisions = {}

    # パンチ判定を考慮したhit, hit_noの評価
    for name, idx in [("hit", 17), ("hit_no", 16)]:
        target_count = sum(targets[:, idx] == 1)
        pred_count = sum((punch_index) & (preds[:, idx] >= threshold))
        correct_count = sum((punch_index) & (preds[:, idx] >= threshold) & (targets[:, idx] == 1))

        recall = correct_count / target_count
        precision = correct_count / (pred_count + 1e-12)

        recalls[f"{name}_with_punch"] = recall
        precisions[f"{name}_with_punch"] = precision

    # パンチ判定とヒット判定を考慮した効果判定の評価
    for name, idx in [("eff_no", 40), ("eff_mild", 41), ("eff_full", 42)]:
        target_count = sum(targets[:, idx] == 1)
        pred_count = sum(
            (punch_index) & (hit_index) & (hit_larger_index) & (preds[:, idx] >= threshold)
        )
        correct_count = sum(
            (punch_index)
            & (hit_index)
            & (hit_larger_index)
            & (preds[:, idx] >= threshold)
            & (targets[:, idx] == 1)
        )

        recall = correct_count / target_count
        precision = correct_count / (pred_count + 1e-12)

        recalls[f"{name}_with_conditions"] = recall
        precisions[f"{name}_with_conditions"] = precision

    return recalls, precisions


def save_metrics_log(
    output_path: str,
    thresholds: list[float],
    recalls: list[dict[str, float]],
    precisions: list[dict[str, float]],
) -> None:
    """評価指標のログを保存する.

    Args:
        output_path: 出力ファイルパス
        thresholds: 閾値のリスト
        recalls: リコールの辞書のリスト
        precisions: 適合率の辞書のリスト
    """
    class_names = ["punch", "hit", "hit_no", "eff_no", "eff_mild", "eff_full"]
    header = "threshold," + ",".join(
        [f"{name}_recall,{name}_precision" for name in class_names]
    )

    with Path(output_path).open("w") as f:
        f.write(f"{header}\n")

        for th, recall_dict, precision_dict in zip(thresholds, recalls, precisions, strict=True):
            values = [th]
            for name in class_names:
                values.extend([recall_dict[name], precision_dict[name]])
            txt = ",".join(f"{v:.4f}" for v in values)
            f.write(f"{txt}\n")
