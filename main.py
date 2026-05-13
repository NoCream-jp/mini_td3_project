# main.py
import datetime
import os
import csv
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import config  # 変数ファイルをインポート
from my_jammer_env import MyJammerEnv  # 環境をインポート

# --- カスタムコールバック ---
class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, total_episodes: int, verbose=0):
        super().__init__(verbose)
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.current_ep_reward = 0.0
        self.episode_rewards = []

    def _on_step(self) -> bool:
        self.current_ep_reward += self.locals["rewards"][0]
        done = self.locals["dones"][0]
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_ep_reward)
            self.current_ep_reward = 0.0 
            
            if self.episode_count >= self.total_episodes:
                return False
        return True

# --- 各種関数 ---
def learn_td3(env):
    model = TD3("MlpPolicy", env, verbose=1)
    callback = EpisodeLoggerCallback(total_episodes=config.TOTAL_EPISODES)
    
    max_possible_timesteps = config.TOTAL_EPISODES * config.MAX_STEPS_PER_EPISODE
    model.learn(total_timesteps=max_possible_timesteps, callback=callback)
    
    model.save(os.path.join(config.OUTPUT_DIR, "simple_td3_model"))
    return model, callback.episode_rewards

def draw_score(now_time, rewards):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, color='green', linewidth=1.5, label='Episode Reward')
    plt.title(f"Learning Curve ({now_time})")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    img_path = os.path.join(config.OUTPUT_DIR, f"score_{now_time}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"学習スコアの画像を保存しました: {img_path}")

def save_result(now_time, model, env):
    with open(os.path.join(config.OUTPUT_DIR, f"test_{now_time}_log.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "agent_x", "agent_y", "jammer_x", "jammer_y"])
        
        obs, _ = env.reset()
        start_pos = np.array(config.AGENT_START_POS, dtype=np.float32)
        env.unwrapped.location = start_pos
        obs[0], obs[1] = start_pos[0], start_pos[1]
        
        for i in range(config.MAX_STEPS_PER_EPISODE):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, finish_flag, over_step_flag, _ = env.step(action)
            
            writer.writerow([i, obs[0], obs[1], obs[2], obs[3]]) 
            if finish_flag or over_step_flag:
                break

def draw_from_csv(now_time):
    csv_path = os.path.join(config.OUTPUT_DIR, f"test_{now_time}_log.csv")

    x_history, y_history = [], []
    jammer_x_history, jammer_y_history = [], []
    
    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            x_history.append(float(row[1]))
            y_history.append(float(row[2]))
            jammer_x_history.append(float(row[3]))
            jammer_y_history.append(float(row[4]))

    plt.figure(figsize=(6, 6))
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)

    plt.scatter(0, 0, color='red', marker='*', s=200, label='Goal (0,0)')

    plt.plot(jammer_x_history, jammer_y_history, color='orange', linestyle='--', linewidth=2.0, label='Jammer Trajectory')
    
    last_jx, last_jy = jammer_x_history[-1], jammer_y_history[-1]
    obstacle_circle = patches.Circle((last_jx, last_jy), radius=config.OBSTACLE_RADIUS, color='grey', alpha=0.5, label='Jammer Final Pos')
    plt.gca().add_patch(obstacle_circle)

    plt.plot(x_history, y_history, color='blue', marker='.', linestyle='-', linewidth=1.5, label='Agent Trajectory')
    plt.scatter(x_history[0], y_history[0], color='green', marker='o', s=100, label='Start')
    
    plt.title(f"Dynamic Jammer Evasion ({now_time})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    img_path = os.path.join(config.OUTPUT_DIR, f"trajectory_{now_time}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"軌跡の画像を保存しました: {img_path}")

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    env = MyJammerEnv()
    
    model, rewards_history = learn_td3(env)
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    draw_score(now_time, rewards_history)
    save_result(now_time, model, env)
    draw_from_csv(now_time)

if __name__ == "__main__":
    main()