"""Precision-Recall AUCを計算するスクリプト."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger

from utils.eval_utils import calculate_prauc


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
        recall_col = f"{class_name}_recall"
        precision_col = f"{class_name}_precision"
        recalls = df[recall_col].values
        precisions = df[precision_col].values
        pr_auc = calculate_prauc(recalls, precisions)
        results[class_name] = pr_auc

    # 結果を表示
    for class_name, pr_auc in results.items():
        logger.info(f"{class_name}: PR-AUC = {pr_auc:.4f}")

    # 全クラスの平均PR-AUC
    pr_aucs = list(results.values())
    logger.info(f"mean PR-AUC: {np.mean(pr_aucs):.4f}")

    # punch, eff_no, eff_mild, eff_fullの平均PR-AUC
    selected_pr_aucs = [pr_aucs[0], pr_aucs[3], pr_aucs[4], pr_aucs[5]]
    logger.info(f"mean PR-AUC w/o hit: {np.mean(selected_pr_aucs):.4f}")


if __name__ == "__main__":
    main()
