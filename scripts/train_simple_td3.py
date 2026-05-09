import datetime
import os
import csv
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 先ほど作った新しい環境をインポート
from jammer_nav.envs.jammer_nav_env import MyJammerEnv

OUTPUT_DIR = "outputs"
TOTAL_TIMESTEPS = 50000

def learn_td3(env):
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(OUTPUT_DIR, "simple_td3_model"))
    return model

def save_result(now_time, model, env):
    with open(os.path.join(OUTPUT_DIR, f"test_{now_time}_log.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "agent_x", "agent_y", "jammer_x", "jammer_y"])
        
        obs, _ = env.reset()
        
        # テスト時のみスタート地点を強制上書き
        start_pos = np.array([1.9, 1.9], dtype=np.float32)
        env.unwrapped.location = start_pos
        # 新しいobsに合わせて更新（[自分のx, 自分のy, ジャマーのx, ジャマーのy]）
        obs[0] = start_pos[0]
        obs[1] = start_pos[1]
        
        for i in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, finish_flag, over_step_flag, _ = env.step(action)
            
            # obsから各座標を取り出して記録
            ax, ay, jx, jy = obs[0], obs[1], obs[2], obs[3]
            writer.writerow([i, ax, ay, jx, jy]) 
            
            if finish_flag or over_step_flag:
                break

def draw_from_csv(now_time):
    csv_path = os.path.join(OUTPUT_DIR, f"test_{now_time}_log.csv")

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

    # ジャマーの軌跡をオレンジの点線でプロット！
    plt.plot(jammer_x_history, jammer_y_history, color='orange', linestyle='--', linewidth=2.0, label='Jammer Trajectory')
    
    # 衝突判定の円は、ジャマーの「最後の位置」に描画する
    last_jx, last_jy = jammer_x_history[-1], jammer_y_history[-1]
    obstacle_circle = patches.Circle((last_jx, last_jy), radius=0.5, color='grey', alpha=0.5, label='Jammer Final Pos')
    plt.gca().add_patch(obstacle_circle)

    # エージェントの軌跡
    plt.plot(x_history, y_history, color='blue', marker='.', linestyle='-', linewidth=1.5, label='Agent Trajectory')
    plt.scatter(x_history[0], y_history[0], color='green', marker='o', s=100, label='Start')
    
    plt.title(f"Dynamic Jammer Evasion ({now_time})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    img_path = os.path.join(OUTPUT_DIR, f"trajectory_{now_time}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"軌跡の画像を保存しました: {img_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = MyJammerEnv()
    model = learn_td3(env)
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_result(now_time, model, env)
    draw_from_csv(now_time)

if __name__ == "__main__":
    main()