# ComfyUI-Janus-Pro

[English](README_EN.md) | [简体中文](README.md) | 日本語

ComfyUI の Janus-Pro ノード、統一されたマルチモーダル理解と生成フレームワーク。

![alt text](<workflow/ComfyUI Janus-Pro-workflow.png>)

## インストール方法

### 方法1: ComfyUI Manager を使用してインストール (推奨)
1. [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) をインストール
2. マネージャーで "Janus-Pro" を検索
3. インストールをクリック

### 方法2: 手動インストール
1. このリポジトリを ComfyUI の custom_nodes フォルダにクローンします:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro
```

2. 必要な依存関係をインストールします:

Windowsの場合:
```bash
# ComfyUI ポータブル版を使用している場合
cd ComfyUI-Janus-Pro
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt

# 独自の Python 環境を使用している場合
cd ComfyUI-Janus-Pro
path\to\your\python.exe -m pip install -r requirements.txt
```

Linux/Macの場合:
```bash
# ComfyUI の Python 環境を使用
cd ComfyUI-Janus-Pro
../../python_embeded/bin/python -m pip install -r requirements.txt

# または独自の環境を使用
cd ComfyUI-Janus-Pro
python -m pip install -r requirements.txt
```

注意: インストールに問題がある場合:
- git がインストールされていることを確認
- pip を更新してみてください: `python -m pip install --upgrade pip`
- プロキシを使用している場合、git が GitHub にアクセスできることを確認
- ComfyUI と同じ Python 環境を使用していることを確認


## モデルのダウンロード

モデルを `ComfyUI/models/Janus-Pro` フォルダに配置します:
1. ComfyUI の models ディレクトリに `Janus-Pro` フォルダを作成
2. Hugging Face からモデルをダウンロード:
   - [Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
   - [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
3. モデルをそれぞれのフォルダに解凍します:
   ```
   ComfyUI/models/Janus-Pro/Janus-Pro-1B/
   ComfyUI/models/Janus-Pro/Janus-Pro-7B/
   ```
