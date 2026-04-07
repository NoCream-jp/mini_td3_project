import os
import csv
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3

# --- 定数（UPPER_SNAKE_CASE） ---
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "simple_td3_model")
LOG_PATH = os.path.join(OUTPUT_DIR, "trajectory_log.csv")
TOTAL_TIMESTEPS = 1000

# --- クラス定義（PascalCase） ---
class SimpleGoalEnv(gym.Env):
    """
    スタート地点ランダムな座標から，(0, 0)を目指す
    """
    def __init__(self):
        super().__init__()
        # 観測空間：自身のXY座標（-2.0 ～ 2.0）
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(2,), dtype=np.float32
        )
        # 行動空間：XY方向への移動量（-0.1 ～ 0.1）
        self.action_space = gym.spaces.Box(
            low=-0.1, high=0.1, shape=(2,), dtype=np.float32
        )
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)
        return self.current_state, {}

    def step(self, action):
        # 状態の更新（変数名は snake_case）
        self.current_state += action
        
        # 報酬計算：ゴールとのユークリッド距離の反転
        distance_to_goal = np.linalg.norm(self.current_state - 1.0)
        reward = -distance_to_goal
        
        # 終了判定：ゴールから 0.1 以内に到達
        done = bool(distance_to_goal < 0.1)
        
        return self.current_state, reward, done, False, {}

# --- メイン処理 ---
def main():
    # 保存先フォルダの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 環境とモデルの初期化
    env = SimpleGoalEnv()
    model = TD3("MlpPolicy", env, verbose=1)

    # 2. 学習の実行
    print(f"学習を開始します（計 {TOTAL_TIMESTEPS} ステップ）...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 3. モデルの保存
    model.save(MODEL_PATH)
    print(f"モデルを保存しました: {MODEL_PATH}")

    # 4. テスト走行とログ記録
    print(f"テスト走行を記録中: {LOG_PATH}")
    with open(LOG_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "x", "y"])

        obs, _ = env.reset()
        for step in range(200):
            # 決定論的な行動を選択
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            writer.writerow([step, obs[0], obs[1]])
            if done:
                print(f"ゴールに到達しました（ステップ: {step}）")
                break

if __name__ == "__main__":
    main()