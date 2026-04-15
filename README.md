# TD3 カスタム環境・API リファレンス

TD3（Twin Delayed DDPG）を用いて独自の連続値制御環境を構築・学習するための最小構成リファレンス．

## 1. 環境構築 (Gymnasium)

TD3は**観測（入力）と行動（出力）の両方が連続値**．<br>
空間定義には必ず `spaces.Box` を使用する．

### 必須実装メソッド (`gym.Env` を継承)

#### ① `__init__(self)`
エージェントの入出力の形状と範囲を定義する．

```python
from gymnasium import spaces
import numpy as np

# 観測空間 (AIが受け取る情報)
self.observation_space = spaces.Box(low=最小値, high=最大値, shape=(次元数,), dtype=np.float32)

# 行動空間 (AIが出力する操作)
self.action_space = spaces.Box(low=最小値, high=最大値, shape=(次元数,), dtype=np.float32)
```

#### ② `reset(self, seed=None, options=None)`
エピソード開始時に環境を初期状態に戻す．

* **Return (必須2要素):**
  1. `obs` (`np.ndarray`): 初期状態の観測データ
  2. `info` (`dict`): 追加情報（基本は空の辞書 `{}` で可）

#### ③ `step(self, action)`
AIからの行動（`action`）を受け取り、環境を1ステップ進める．

* **Return (必須5要素):**
  1. `obs` (`np.ndarray`): 更新後の観測データ
  2. `reward` (`float`): 行動に対する報酬
  3. `terminated` (`bool`): エピソードの終了判定（成功・衝突など）
  4. `truncated` (`bool`): 制限時間（最大ステップ数）による打ち切り判定
  5. `info` (`dict`): デバッグ用情報（基本は `{}` で可）

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