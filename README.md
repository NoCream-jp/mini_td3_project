---

## 学習・推論 (Stable Baselines3)

作成したカスタム環境でTD3エージェントを動かすための基本コードである。本プロジェクトでは、学習回数の管理や記録のためにカスタムコールバックも併用している。

```python
from stable_baselines3 import TD3
from my_jammer_env import MyJammerEnv
import config

# 1. 環境の初期化 (自作環境のインスタンス化)
env = MyJammerEnv()

# 2. モデルの定義と学習
# MlpPolicy: 多層パーセプトロン（標準的なニューラルネットワーク）
model = TD3("MlpPolicy", env, verbose=1)
# 実際にはコールバックを用いて、指定エピソード数（TOTAL_EPISODES）で学習を停止させる
model.learn(total_timesteps=config.TOTAL_EPISODES * config.MAX_STEPS_PER_EPISODE)

# 3. モデルの保存と読み込み
model.save("outputs/simple_td3_model")
loaded_model = TD3.load("outputs/simple_td3_model")

# 4. テスト走行（推論）
obs, _ = env.reset()
# deterministic=True: 確率的要素を排除し、学習済みの最適な行動を選択
action, _states = model.predict(obs, deterministic=True)
```

---

## 利用しているパラメータの説明

本プロジェクトのコードにおいて、制御の要となる主要な変数とその役割である。現在は `config.py` に設定を集約している。

### システム・学習設定 (`config.py`)
* **`OUTPUT_DIR`**
  * **役目:** 学習済みのモデルデータや、テスト走行の結果（CSV、グラフ画像）を保存するための出力先ディレクトリ。
* **`TOTAL_EPISODES`**
  * **役目:** AIが環境内で目標達成（または失敗）するまでの一連の行動を「1回」としたとき、それを何回繰り返して学習するか（総エピソード数）。
* **`MAX_STEPS_PER_EPISODE`**
  * **役目:** 1エピソードの中でエージェントが行動できる最大歩数。この歩数に達すると時間切れ（引き分け）としてエピソードがリセットされる。

### 環境パラメータ (`config.py` / `MyJammerEnv`)
* **`AGENT_START_POS` / `JAMMER_START_POS`**
  * **役目:** テスト走行時における、エージェント（AI）とジャマー（動的障害物）の初期配置座標。
* **`JAMMER_SPEED`**
  * **役目:** ジャマーが毎ステップ移動する速度。この値を持たせることで、単なる静的障害物から「動的障害物」へと環境が進化している。
* **`OBSTACLE_RADIUS` / `GOAL_TOLERANCE`**
  * **役目:** 障害物との衝突を判定する半径と、ゴール到達を判定する許容距離。

### AIの入出力と状態
* **`self.location` / `self.jam`**
  * **役目:** エージェントとジャマーの「現在の物理的な座標・状態」。環境内部の真のステータスとして毎ステップ更新される。
* **`obs` (観測状態)**
  * **役目:** エージェントが行動を決めるために環境から受け取る「センサー情報」。本環境では `[自分のx, 自分のy, ジャマーのx, ジャマーのy]` の4つの数値が渡され、AIは動く障害物の位置を認識する。
* **`action`**
  * **役目:** 学習済みのAIモデルが決定した「次の行動」。X軸・Y軸の移動量（`-0.1` ～ `+0.1`）として環境に出力される。
* **`reward`**
  * **役目:** AIへの「報酬（点数）」。ゴール到達で特大ボーナス（+1000）、障害物衝突で即死ペナルティ（-1000）、移動中はゴールまでの距離に応じたマイナス点が与えられ、AIはこの合計スコアの最大化を目指す。

---

### venv環境立ち上げ
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### venv上でのライブラリインストールと実行
```cmd
pip install stable-baselines3 gymnasium numpy matplotlib
python main.py
```

### venv終了
```cmd
deactivate
```

---

## プロジェクト構成
役割ごとにファイルを分割した

```text
mini_td3_project/
├── config.py               # 実験で変更するパラメータや設定を管理する変数ファイル
├── my_jammer_env.py        # 動的障害物（ジャマー）を含む自作の環境クラス
├── main.py                 # 学習の実行、テスト走行、グラフ描画を統括するメインスクリプト
├── outputs/                # プログラム実行により自動生成される成果物
│   ├── simple_td3_model.zip# 学習済みモデル（ニューラルネットワークの重み）
│   ├── test_YYYY...csv     # テスト走行時のエージェントとジャマーの座標ログ
│   ├── score_YYYY...png    # 学習曲線（エピソードごとの累積報酬の推移グラフ）
│   └── trajectory_YYYY...png # テスト走行の軌跡と、ジャマーの動きを可視化した画像
├── .gitignore              # Git管理から除外するファイルの設定
└── README.md               # 本ファイル（プロジェクトの説明書）
```