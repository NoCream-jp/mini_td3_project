import datetime
import os
import csv
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import config
from my_jammer_env import MyJammerEnv

# ノイズインポート
from stable_baselines3.common.noise import NormalActionNoise

# コールバック関数
## データロガー
class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, total_episodes: int, verbose=0):
        super().__init__(verbose)
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.current_ep_reward = 0.0
        self.episode_rewards = []

    # 報酬を描画用にrewardsに保存
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

def learn_td3(env):    
    # ノイズ設定(ランダム性を持たせる設定)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # モデル用意
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    # コールバック用意
    callback = EpisodeLoggerCallback(total_episodes=config.TOTAL_EPISODES)
    # タイムステップ上限設定
    max_possible_timesteps = config.TOTAL_EPISODES * config.MAX_STEPS_PER_EPISODE
    # learn打つ
    model.learn(total_timesteps=max_possible_timesteps, callback=callback)
    # save
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

# テストエピソードを回すために呼ばれる関数
"""
- csvに保存(csvファイルには本番テスト結果しか入っていない)
- reset()
- config.MAX_STEPS_PER_EPISODEまたは終了条件まで繰り返しstep()
- 
"""
def save_result(now_time, model, env):
    num_jammers = env.unwrapped.num_jammers
    with open(os.path.join(config.OUTPUT_DIR, f"test_{now_time}_log.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        
        # ジャマーの数に合わせてCSVのヘッダーを動的に生成
        header = ["step", "agent_x", "agent_y"]
        for i in range(num_jammers):
            header.extend([f"j{i}_x", f"j{i}_y"])
        writer.writerow(header)
        
        obs, _ = env.reset()
        start_pos = np.array(config.AGENT_START_POS, dtype=np.float32)
        env.unwrapped.location = start_pos
        obs[0], obs[1] = start_pos[0], start_pos[1]
        
        for i in range(config.MAX_STEPS_PER_EPISODE):
            # 次の動きをmodelから自動計算
            ## deterministic = Trueで決定論的な動きに設定できる(ノイズを除去できる)
            action, _ = model.predict(obs, deterministic=True)
            # stepを進める
            obs, _, finish_flag, over_step_flag, _ = env.step(action)
            
            # obsからデータを抽出して行を作成
            row_data = [i, obs[0], obs[1]]
            for j in range(num_jammers):
                # 0,1はエージェント。ジャマーのx,yは2から2個ずつ格納されている
                row_data.extend([obs[2 + j*2], obs[3 + j*2]])
            
            writer.writerow(row_data) 
            if finish_flag or over_step_flag:
                break

# 最後のテスト試行で生成したcsvから描画する関数
def draw_from_csv(now_time):
    csv_path = os.path.join(config.OUTPUT_DIR, f"test_{now_time}_log.csv")

    x_history, y_history = [], []
    jammer_histories = {} 
    
    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        if len(header) < 3:
            raise ValueError(f"Invalid CSV header in {csv_path}")
        # ヘッダーの列数からジャマーの数を逆算
        num_jammers = (len(header) - 3) // 2

        for i in range(num_jammers):
            jammer_histories[i] = {'x': [], 'y': []}

        for row in reader:
            if len(row) < 3 + num_jammers * 2:
                continue
            x_history.append(float(row[1]))
            y_history.append(float(row[2]))
            for i in range(num_jammers):
                jammer_histories[i]['x'].append(float(row[3 + i*2]))
                jammer_histories[i]['y'].append(float(row[4 + i*2]))

    if not x_history:
        raise ValueError(f"No trajectory data in {csv_path}")

    plt.figure(figsize=(6, 6))
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    plt.scatter(0, 0, color='red', marker='*', s=200, label='Goal (0,0)')

    # ジャマーの描画（複数いるため色を変えてプロット）
    colors = ['orange', 'purple', 'cyan', 'brown', 'pink']
    for i in range(num_jammers):
        c = colors[i % len(colors)]
        jx_hist = jammer_histories[i]['x']
        jy_hist = jammer_histories[i]['y']
        
        plt.plot(jx_hist, jy_hist, color=c, linestyle='--', linewidth=2.0, label=f'Jammer {i+1} Traj')
        last_jx, last_jy = jx_hist[-1], jy_hist[-1]
        obstacle_circle = patches.Circle((last_jx, last_jy), radius=config.OBSTACLE_RADIUS, color='grey', alpha=0.5)
        plt.gca().add_patch(obstacle_circle)

    plt.plot(x_history, y_history, color='blue', marker='.', linestyle='-', linewidth=1.5, label='Agent Trajectory')
    plt.scatter(x_history[0], y_history[0], color='green', marker='o', s=100, label='Start')
    
    plt.title(f"Dynamic Jammer Evasion ({now_time})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 凡例がグラフに被らないように外側に配置
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    
    img_path = os.path.join(config.OUTPUT_DIR, f"trajectory_{now_time}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"軌跡の画像を保存しました: {img_path}")


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 環境をインスタンス化
    env = MyJammerEnv()
    
    # 学習
    model, rewards_history = learn_td3(env)
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # 描画
    draw_score(now_time, rewards_history)
    save_result(now_time, model, env)
    draw_from_csv(now_time)



if __name__ == "__main__":
    main()