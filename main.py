import datetime
import os
import csv
import numpy as np
import copy
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 自作ファイルインポート
import config
from my_jammer_env import MyJammerEnv
from my_wrappers import TrajectoryPredictionWrapper, SafetyShieldWrapper, VelocityObservationWrapper

# ノイズインポート
from stable_baselines3.common.noise import NormalActionNoise

# ラッパーインポート
from my_wrappers import (
    TrajectoryPredictionWrapper, 
    SafetyShieldWrapper, 
    VelocityObservationWrapper,
    KalmanPredictionWrapper,
    PotentialFieldShieldWrapper  # 🆕 これを追加！
)

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
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
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

# テストエピソード(1周だけ)を回し、記録するために呼ばれる関数
def actual_test(now_time, model, env):
    num_jammers = env.unwrapped.num_jammers
    prediction_snapshots = []
    
    with open(os.path.join(config.OUTPUT_DIR, f"test_{now_time}_log.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        
        header = ["step", "agent_x", "agent_y"]
        for i in range(num_jammers):
            header.extend([f"j{i}_x", f"j{i}_y"])
        writer.writerow(header)
        
        # 修正：無理な上書きをすべて廃止し、この1行だけで完璧に同期させて初期化する
        obs, info = env.reset(options={"start_pos": config.AGENT_START_POS})
        
        for i in range(config.MAX_STEPS_PER_EPISODE):
            action, _ = model.predict(obs, deterministic=True)
            
            # 30ステップごとに、純粋な数値を複製してメモリに保存
            if i % 30 == 0 and 'jam_preds' in info:
                agent_pos = (env.unwrapped.location[0], env.unwrapped.location[1])
                prediction_snapshots.append({
                    "step": i,
                    "agent_pos": agent_pos,
                    "preds": copy.deepcopy(info['jam_preds'])
                })
            
            # stepを進める
            obs, _, finish_flag, over_step_flag, info = env.step(action)
            
            # obsからデータを抽出して行を作成
            row_data = [i, obs[0], obs[1]]
            for j in range(num_jammers):
                row_data.extend([obs[2 + j*2], obs[3 + j*2]])
            
            writer.writerow(row_data) 
            
            # ここで激突（finish_flag）したら、即座にループを抜けてログ保存を終了する
            if finish_flag or over_step_flag:
                print(f"★本番テスト：ステップ {i} で衝突判定、または終了条件を検知しました。")
                break
                
    return prediction_snapshots

# 最後のテスト試行で生成したcsvから描画する関数
def draw_from_csv(now_time, prediction_snapshots=None):
    csv_path = os.path.join(config.OUTPUT_DIR, f"test_{now_time}_log.csv")

    x_history, y_history = [], []
    jammer_histories = {} 
    
    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        if len(header) < 3:
            raise ValueError(f"Invalid CSV header in {csv_path}")
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

    plt.figure(figsize=(8, 6))
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    plt.gca().set_aspect('equal')
    gx, gy = config.GOAL_POS
    plt.scatter(gx, gy, color='red', marker='*', s=200, label=f'Goal ({gx}, {gy})', zorder=5)

    # ジャマーの描画
    colors = ['orange', 'purple', 'cyan', 'brown', 'pink']
    for i in range(num_jammers):
        c = colors[i % len(colors)]
        jx_hist = jammer_histories[i]['x']
        jy_hist = jammer_histories[i]['y']
        
        plt.plot(jx_hist, jy_hist, color=c, linestyle='--', linewidth=2.0, label=f'Jammer {i+1} Traj')
        last_jx, last_jy = jx_hist[-1], jy_hist[-1]
        obstacle_circle = patches.Circle((last_jx, last_jy), radius=config.OBSTACLE_RADIUS, color='grey', alpha=0.5, zorder=3)
        plt.gca().add_patch(obstacle_circle)

    # エージェントの軌跡描画
    plt.plot(x_history, y_history, color='blue', marker='.', linestyle='-', linewidth=1.5, label='Agent Trajectory', zorder=4)
    plt.scatter(x_history[0], y_history[0], color='green', marker='o', s=100, label='Start', zorder=5)
    
    # 引数で直接受け取ったメモリ上の予測リストを展開して描画
    # if prediction_snapshots is not None:
    #     for idx, shot in enumerate(prediction_snapshots):
    #         a_pos = shot["agent_pos"]
    #         all_jam_preds = shot["preds"]
            
    #         # 予測が行われた位置に小さな黒丸を打つ
    #         plt.scatter(a_pos[0], a_pos[1], color='black', marker='o', s=25, zorder=5)
            
    #         # 予測軌道を描画
    #         for jam_idx, pred_traj in enumerate(all_jam_preds):
    #             px = [pt[0] for pt in pred_traj]
    #             py = [pt[1] for pt in pred_traj]
                
    #             label = "Jammer Prediction" if idx == 0 and jam_idx == 0 else ""
    #             plt.plot(px, py, color='darkorange', linestyle=':', alpha=0.7, linewidth=1.8, label=label, zorder=2)

    plt.title(f"Dynamic Jammer Evasion ({now_time})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 凡例を外側に配置
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    
    img_path = os.path.join(config.OUTPUT_DIR, f"trajectory_{now_time}.png")
    plt.savefig(img_path) 
    plt.close()
    print(f"軌跡の画像を保存しました: {img_path}")


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    #---------------wrapper装備----------------------
    # 0. まず生の環境を作成
    raw_env = MyJammerEnv()

    # ==========================================
    # 実験スイッチ：実行したい手法を1つだけ選び、コメントを外してください
    # ==========================================

    # 【手法1】純粋な強化学習（座標のみ）
    # env = raw_env

    # 【手法2】強化学習 ＋ 予測シールド（CVモデル: 等速直線）
    # env = TrajectoryPredictionWrapper(raw_env, history_length=2, horizon_steps=20)
    # env = SafetyShieldWrapper(env, lookahead_steps=15, safety_margin=0.35)

    # 【手法3】強化学習 ＋ 速度ベクトル入力
    # env = VelocityObservationWrapper(raw_env)

    # 【手法4】ハイブリッド（速度入力 ＋ CV予測シールド）
    # env = VelocityObservationWrapper(raw_env)
    # env = TrajectoryPredictionWrapper(env, history_length=2, horizon_steps=20)
    # env = SafetyShieldWrapper(env, lookahead_steps=15, safety_margin=0.35)

    # 【手法5】最新ハイブリッド（速度入力 ＋ カルマンフィルタ予測シールド）
    env = VelocityObservationWrapper(raw_env)
    env = KalmanPredictionWrapper(env, horizon_steps=20)
    env = SafetyShieldWrapper(env, lookahead_steps=15, safety_margin=0.35)

    # 【手法6】カルマン予測 ＋ 人工ポテンシャル法シールド（APF）
    env = VelocityObservationWrapper(raw_env)
    env = KalmanPredictionWrapper(env, horizon_steps=20)
    # 古いシールドの代わりに APF シールドを被せる
    env = PotentialFieldShieldWrapper(env, lookahead_steps=15, safety_margin=0.35, k_rep=0.05)
    #------------------------------------------------

    # 環境envを利用して学習を実行する
    model, rewards_history = learn_td3(env)
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # 学習時のスコアの描画
    draw_score(now_time, rewards_history)
    
    # actual_test から予測リストを受け取る
    pred_snapshots = actual_test(now_time, model, env)
    
    # 受け取ったリストをそのまま draw_from_csv に引き渡す
    draw_from_csv(now_time, pred_snapshots)


if __name__ == "__main__":
    main()