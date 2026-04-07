import os
import csv
import numpy as np
# 環境
import gymnasium as gym
# 学習
from stable_baselines3 import TD3
# --- 定数 ---

OUTPUT_DIR = "outputs"
TOTAL_TIMESTEPS = 1000 # 1000ステップでも回る

"""
ランダムな場所から原点を目指す
"""

class RandomStartEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self.state = np.zeros(2, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # スタート地点を -2.0 ～ 2.0 の間でランダムに決定
        self.state = self.np_random.uniform(low=-2.0, high=2.0, size=(2,)).astype(np.float32)
        return self.state, {}

    def step(self, action):
        self.state += action
        # 報酬：原点 (0, 0) との距離が近いほど高い
        dist = np.linalg.norm(self.state)
        reward = -dist
        done = bool(dist < 0.1)
        return self.state, reward, done, False, {}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import time
    env = RandomStartEnv()
    
    # 学習
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(OUTPUT_DIR, "simple_td3_model"))

    # テスト走行を1回分記録
    with open(os.path.join(OUTPUT_DIR, f"test_{int(time.time())}_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y"])
        obs, _ = env.reset()
        for i in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            writer.writerow([i, obs[0], obs[1]])
            if done: break

if __name__ == "__main__":
    main()