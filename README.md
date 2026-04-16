

---

## 2. 学習・推論 (Stable Baselines3)

作成したカスタム環境でTD3エージェントを動かすための基本コード．

```python
from stable_baselines3 import TD3

# 1. 環境の初期化 (自作環境のインスタンス化)
env = YourCustomEnv()

# 2. モデルの定義と学習
# MlpPolicy: 多層パーセプトロン（標準的なニューラルネットワーク）
model = TD3("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 3. モデルの保存と読み込み
model.save("td3_model")
loaded_model = TD3.load("td3_model")

# 4. テスト走行（推論）
obs, _ = env.reset()
# deterministic=True: 確率的要素を排除し、学習済みの最適な行動を選択
action, _states = model.predict(obs, deterministic=True)

```

---


## 利用しているパラメータの説明

本プロジェクトのコードにおいて，制御の要となる主要な変数とその役割である．設定したカスタム変数名に基づいている．

### システム・学習設定
* **`OUTPUT_DIR`**
  * **役目:** 学習済みのモデルデータや，テスト走行の結果を記録したCSVファイルを保存するための出力先ディレクトリ名．
* **`TOTAL_STEPS`**
  * **役目:** AIが環境内で行動を繰り返して学習する「総ステップ数（練習の回数）」．

### 環境内部のステータス
* **`self.location`**
  * **役目:** エージェント（AI）の「現在の物理的な座標」．`reset` 時にランダムな初期位置が代入され，`step` ごとに物理的に更新される環境内の真の状態．
* **`dist`**
  * **役目:** 現在地（`self.location`）から目標地点（原点）までの直線距離．報酬計算やゴール判定の絶対的な基準となる．
* **`reward`**
  * **役目:** AIへの「報酬（点数）」．目標に近づくほどマイナスが小さく（＝高く）なるよう設定され，AIはこの数値を最大化するように学習する．
* **`done_flag`**
  * **役目:** 1エピソードの終了判定フラグ．目標に十分近づいた（`dist` が規定値未満になった）場合に `True` となり，シミュレーションのクリアを示す．

### AIの入出力
* **`action`**
  * **役目:** 学習済みのAIモデルが現在の状況を判断して決定した「次の行動（操作コマンド）」．環境の `step` 関数に入力され，状態を変化させる．
* **`observed_state`**
  * **役目:** エージェントが行動を決めるために環境から受け取る「センサーからの観測情報」．`reset` や `step` から出力され，AIの次の予測に利用される．


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