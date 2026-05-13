import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import TD3
import os

# ==========================================
# 0. 定数設定
# ==========================================
OUTPUT_DIR = "outputs"
TOTAL_STEPS = 10000  # 学習の総ステップ数

# ==========================================
# 1. カスタム環境の定義
# ==========================================
class RandomStartEnv(gym.Env):
    """
    ランダムな初期位置から原点(0,0)を目指すカスタム環境
    """
    def __init__(self):
        super().__init__()

        # 【必須】空間の定義
        # 行動空間: x, y方向への移動量 [-0.1 ~ 0.1]
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

        # 観測空間: エージェントの現在座標 [-2.0 ~ 2.0]
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)

        # 内部変数（環境の真の状態）
        self.state = np.zeros(2, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        環境の初期化
        """
        super().reset(seed=seed)

        # スタート地点を -2.0 ～ 2.0 の間でランダムに決定
        self.state = self.np_random.uniform(low=-2.0, high=2.0, size=(2,)).astype(np.float32)

        # 初期観測状態を生成
        observed_state = self.state.copy()
        
        return observed_state, {}

    def step(self, action):
        """
        環境の更新（1ステップ進める）
        """
        # 1. 行動に従ってエージェントの内部状態を更新
        self.state += action
        self.state = np.clip(self.state, -2.0, 2.0)

        # 2. 目標地点（原点）までの距離を計算
        dist = np.linalg.norm(self.state)

        # 3. 報酬計算（距離が近いほど報酬が高い）
        reward = -float(dist)

        # 4. 終了判定（距離が0.1未満ならクリア）
        done_flag = bool(dist < 0.1)
        truncated = False # 今回は時間切れ判定なし（必要に応じて実装）

        # 5. 次の観測状態を取得
        observed_state = self.state.copy()

        return observed_state, reward, done_flag, truncated, {}

# ==========================================
# 2. 学習と実行のメイン処理
# ==========================================
def main():
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 環境のインスタンス化
    env = RandomStartEnv()

    # モデルの初期化（TD3）
    model = TD3("MlpPolicy", env, verbose=1)

    # 学習開始
    print(f"学習を開始します（総ステップ数: {TOTAL_STEPS}）")
    model.learn(total_timesteps=TOTAL_STEPS)
    
    # モデルの保存
    model.save(os.path.join(OUTPUT_DIR, "td3_navigation_model"))
    print("学習が完了し，モデルを保存しました．")

    # --- テスト走行（推論） ---
    observed_state, _ = env.reset()
    print("テスト走行を開始します．")
    
    for i in range(200):
        # AIによる行動決定
        action, _states = model.predict(observed_state, deterministic=True)
        
        # 環境の更新
        observed_state, reward, done_flag, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i}: Location={observed_state}, Reward={reward:.4f}")

        if done_flag:
            print(f"目標に到達しました！ (Step: {i})")
            break

if __name__ == "__main__":
    main()