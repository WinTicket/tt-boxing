# パンチ分類モデルのuv移行手順書

## 概要
既存の`punch_classification`プロジェクトをuvを使用したPythonプロジェクトとして再構築する手順を記載します。

## 前提条件
- macOS Sonoma
- Python 3.10以上
- GPUサポート（NVIDIA RTX 3090Ti推奨）

## 移行手順

### 1. プロジェクト構造の作成

```bash
mkdir -p refactor/projects/punch_classification
cd refactor/projects/punch_classification

# 必要なディレクトリの作成
mkdir -p {src,tests,data,model,output}
```

### 2. uv環境のセットアップ

```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトの初期化
uv venv
source .venv/bin/activate

# pyproject.tomlの作成
touch pyproject.toml
```

### 3. pyproject.tomlの設定

```toml
[project]
name = "punch_classification"
version = "0.1.0"
description = "Boxing punch classification model"
requires-python = ">=3.10"
dependencies = [
    "torch==2.5.0",
    "torchvision==0.20.0",
    "torchaudio==2.5.0",
    "numpy==1.26.0",
    "opencv-python",
    "thop",
    "scipy",
    "matplotlib",
    "imageio",
    "tqdm",
    "pandas",
    "scikit-learn"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.ruff]
select = ["E", "F", "I"]
line-length = 88
```

### 4. ファイルの移行

以下のファイルを`src/punch_classification/`に移行：

- `dataset.py` → `src/punch_classification/dataset.py`
- `transforms.py` → `src/punch_classification/transforms.py`
- `scheduler.py` → `src/punch_classification/scheduler.py`
- 学習関連：
  - `train_stack_multiframe.py` → `src/punch_classification/training/train_stack_multiframe.py`
  - `train_stack_multiframe_twostage.py` → `src/punch_classification/training/train_stack_multiframe_twostage.py`
- 評価関連：
  - `eval_prauc.py` → `src/punch_classification/evaluation/eval_prauc.py`
  - `eval_precision_recall_stack_multiframe.py` → `src/punch_classification/evaluation/eval_precision_recall_stack_multiframe.py`
- ユーティリティ：
  - `prepare_squaredata.py` → `src/punch_classification/utils/prepare_squaredata.py`
  - `split_winticket_video.py` → `src/punch_classification/utils/split_winticket_video.py`
  - `export_det_result.py` → `src/punch_classification/utils/export_det_result.py`
  - `visualize_det_result.py` → `src/punch_classification/utils/visualize_det_result.py`

### 5. 依存関係のインストール

```bash
# PyTorchのインストール（CUDA 11.8サポート）
uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

# その他の依存関係のインストール
uv pip install -e .
```

### 6. データとモデルの移行

1. データの移行:
```bash
cp -r ../../../20250226_ボクシング向け画像認識納品物_1/ソフトウェア/punch_classification/data/* data/
```

2. モデルの移行:
```bash
cp -r ../../../20250226_ボクシング向け画像認識納品物_1/ソフトウェア/punch_classification/model/4Q_best/* model/
```

### 7. 実行スクリプトの作成

`scripts`ディレクトリを作成し、主要な操作のためのスクリプトを配置：

```bash
mkdir scripts
```

以下のスクリプトを作成：
- `scripts/train.py` - 学習実行用
- `scripts/evaluate.py` - 評価実行用
- `scripts/prepare_data.py` - データ準備用

### 8. テストの作成

`tests`ディレクトリに単体テストを作成：

```bash
# 主要なモジュールのテストファイル
touch tests/test_dataset.py
touch tests/test_transforms.py
touch tests/test_scheduler.py
```

### 9. ドキュメントの移行と更新

```bash
mkdir docs
cp ../../../20250226_ボクシング向け画像認識納品物_1/ソフトウェア/punch_classification/*.md docs/
```

既存のドキュメントを更新し、新しい環境での実行手順を追加。

### 10. 動作確認

1. 環境のセットアップ:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

2. テストの実行:
```bash
python -m pytest
```

3. 学習の実行:
```bash
python scripts/train.py
```

4. 評価の実行:
```bash
python scripts/evaluate.py
```

## 注意事項

- 既存のコードはそのまま保持し、新しい環境で動作確認を行ってから古い環境を削除すること
- 依存関係に問題が発生した場合は、`requirements.txt`を生成して詳細な依存関係を管理することも検討
- GPUサポートの設定は、システムのCUDAバージョンに応じて適切に調整すること

## 参考資料

- [uv documentation](https://github.com/astral-sh/uv)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)