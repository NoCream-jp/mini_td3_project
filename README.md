

---

### venv環境立ち上げ
```
python -m venv .venv<br>
.venv\Scripts\activate.bat<br>
```

### venv上　ライブラリインストールとスクリプトの実行
```
pip install stable-baseline3 gymnasium numpy
python -m scripts.train_simple_td3
```

### venv終了
```
deactivate
```

## プロジェクト構成
```
mini_td3_project/
├── scripts/                # 実行用スクリプトを格納
│   └── train_simple_td3.py  # 環境定義、学習、評価を一本化したメインプログラム
├── outputs/                # プログラム実行により生成される成果物
│   ├── simple_td3_model.zip # 学習済みモデル（ニューラルネットワークの重み）
│   └── trajectory_log.csv   # テスト走行時のエージェントの座標データ
├── .gitignore              # Git管理から除外するファイルの設定
└── README.md               # 本ファイル（プロジェクトの説明書）
```