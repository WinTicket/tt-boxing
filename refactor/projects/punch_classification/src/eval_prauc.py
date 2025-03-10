"""Precision-Recall AUCを計算するモジュール."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import auc


def calculate_pr_auc(df: pd.DataFrame, class_name: str) -> float:
    """指定されたクラスのPrecision-Recall AUCを計算する.

    Args:
        df: Precision-Recallデータを含むDataFrame
        class_name: 評価対象のクラス名

    Returns:
        PR-AUC値
    """
    recall_col = f"{class_name}_recall"
    precision_col = f"{class_name}_precision"

    # 0のprecisionとrecallを追加
    recalls = np.append(df[recall_col].values, [0])
    precisions = np.append(df[precision_col].values, [1])

    # NaNを0に置換
    recalls = np.nan_to_num(recalls)
    precisions = np.nan_to_num(precisions)

    # リストを降順にソート
    sorted_indices = np.argsort(recalls)[::-1]
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    return auc(recalls, precisions)


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(description="Precision-Recall AUCを計算するスクリプト")
    parser.add_argument("-cp", "--csvpath", type=str, required=True, help="評価対象のCSVファイルパス")
    args = parser.parse_args()

    # CSVファイルを読み込む
    df = pd.read_csv(args.csvpath)

    # 評価対象のクラス
    classes: list[str] = ["punch", "hit", "hit_no", "eff_no", "eff_mild", "eff_full"]

    # 各クラスのPR-AUCを計算
    results: dict[str, float] = {}
    for class_name in classes:
        pr_auc = calculate_pr_auc(df, class_name)
        results[class_name] = pr_auc

    # 結果を表示
    for class_name, pr_auc in results.items():
        print(f"{class_name}: PR-AUC = {pr_auc:.4f}")

    # 全クラスの平均PR-AUC
    pr_aucs = list(results.values())
    print(f"mean PR-AUC: {np.mean(pr_aucs):.4f}")

    # punch, eff_no, eff_mild, eff_fullの平均PR-AUC
    selected_pr_aucs = [pr_aucs[0], pr_aucs[3], pr_aucs[4], pr_aucs[5]]
    print(f"mean PR-AUC w/o hit: {np.mean(selected_pr_aucs):.4f}")


if __name__ == "__main__":
    main()
