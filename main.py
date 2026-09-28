import datetime
import os
import csv
import numpy as np
import copy
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import glob

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
    PotentialFieldShieldWrapper,
    MonteCarloPredictionWrapper
)

# コールバック関数ふくむクラス
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

# 学習を進める
def learn_td3(env, now_time: str):
    # ノイズ設定(ランダム性を持たせる設定)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=np.ones(n_actions) * 0.3)
    # モデル用意
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    # コールバック用意
    callback = EpisodeLoggerCallback(total_episodes=config.TOTAL_EPISODES)
    # タイムステップ上限設定
    max_possible_timesteps = config.TOTAL_EPISODES * config.MAX_STEPS_PER_EPISODE
    # learn打つ
    model.learn(total_timesteps=max_possible_timesteps, callback=callback)
    # save
    model_save_path = os.path.join(config.OUTPUT_DIR, f"{now_time}_simple_td3_model")
    model.save(model_save_path)
    return model, callback.episode_rewards

# 本番テストエピソード(1周だけ)を回し、記録するために呼ばれる関数
def actual_test(now_time, model, env):
    num_jammers = env.unwrapped.num_jammers
    prediction_snapshots = []
    csv_filename = f"{now_time}_test_log.csv"   # ← now_time プレフィックスで統一
    csv_path = os.path.join(config.OUTPUT_DIR, csv_filename)

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        header = ["step", "agent_x", "agent_y"]
        for i in range(num_jammers):
            header.extend([f"j{i}_x", f"j{i}_y"])
        header.append("reward")
        writer.writerow(header)

        obs, info = env.reset(options={"start_pos": config.AGENT_START_POS})

        for i in range(config.MAX_STEPS_PER_EPISODE):
            action, _ = model.predict(obs, deterministic=True)

            if i % 30 == 0 and 'jam_preds' in info:
                agent_pos = (env.unwrapped.location[0], env.unwrapped.location[1])
                prediction_snapshots.append({
                    "step": i,
                    "agent_pos": agent_pos,
                    "preds": copy.deepcopy(info['jam_preds'])
                })

            obs, reward, finish_flag, over_step_flag, info = env.step(action)

            row_data = [i, obs[0], obs[1]]
            for j in range(num_jammers):
                row_data.extend([obs[2 + j*2], obs[3 + j*2]])
            row_data.append(reward)

            writer.writerow(row_data)

            if finish_flag or over_step_flag:
                print(f"本番テスト：ステップ {i} で衝突判定、または終了条件を検知しました。")
                break

    print(f"テストログ CSV を保存しました: {csv_path}")
    return prediction_snapshots
# 報酬可視化関数
def draw_score(now_time, rewards):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, color='green', linewidth=1.5, label='Episode Reward')
    plt.title(f"Learning Curve ({now_time})")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward (Symlog Scale)")
    plt.yscale('symlog', linthresh=100)
    plt.ylim(-2500, 1500)
    plt.axhline(0, color='black', linewidth=1.0, linestyle='--', zorder=1)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    img_path = os.path.join(config.OUTPUT_DIR, f"{now_time}_score.png")
    plt.savefig(img_path)
    plt.close()
    print(f"学習スコアの画像を保存しました: {img_path}")

# 報酬値の移動平均を可視化する関数
def draw_score_moving_average(
    now_time,
    rewards,
    window: int = 100,
    min_periods: int = 1,
    plot_raw: bool = False,
    figsize: tuple = (8, 5),
    dpi: int = 100,
    ylim: tuple = (-2500, 1500),
    ):
    """
    直近 `window` エピソードのトレーリング移動平均を計算してプロット・保存する。

    引数:
      now_time: ファイル名等に用いるタイムスタンプ文字列
      rewards: エピソードごとの報酬を並べた iterable（list / ndarray）
      window: 移動平均のウィンドウ幅（直近 window ステップ: i-window+1 ... i）
      min_periods: 平均を計算する最小サンプル数。例えば 1 にすると先頭から平均を出す。
      plot_raw: 元のエピソード報酬も併せて描画するか。
      figsize, dpi, ylim: 描画パラメータ
    返り値:
      moving_avg (list): 各エピソードに対応する移動平均（条件を満たさない箇所は np.nan）
    """
    # 型変換
    arr = np.array(rewards, dtype=float)
    n = len(arr)
    if n == 0:
        print("draw_score_moving_average: rewards が空です。処理を中止します。")
        return []

    # 移動平均（トレーリングウィンドウ）
    moving_avg = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        if count >= min_periods:
            moving_avg[i] = arr[start : i + 1].mean()

    # プロット
    plt.figure(figsize=figsize, dpi=dpi)
    episodes = np.arange(1, n + 1)

    if plot_raw:
        plt.plot(episodes, arr, color="green", linewidth=1.0, alpha=0.25, label="Episode Reward (raw)")

    plt.plot(episodes, moving_avg, color="orange", linewidth=1.8, label=f"Moving Avg (window={window})")

    plt.title(f"Learning Curve - Moving Average (window={window}) ({now_time})")
    plt.xlabel("Episodes")
    plt.ylabel("Moving Average Reward")
    plt.yscale("symlog", linthresh=100)
    plt.ylim(ylim)
    plt.axhline(0, color="black", linewidth=1.0, linestyle="--", zorder=1)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    # 保存
    img_path = os.path.join(config.OUTPUT_DIR, f"{now_time}_score_ma_w{window}.png")
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()
    print(f"移動平均スコアの画像を保存しました: {img_path}")
    return moving_avg.tolist()

# 最後のテスト試行で生成したcsvから描画する関数
def draw_from_csv(now_time, prediction_snapshots=None):
    csv_path = os.path.join(config.OUTPUT_DIR, f"{now_time}_test_log.csv")

    x_history, y_history = [], []
    jammer_histories = {} 
    
    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        if len(header) < 3:
            raise ValueError(f"Invalid CSV header in {csv_path}")
        num_jammers = (len(header) - 4) // 2

        for i in range(num_jammers):
            jammer_histories[i] = {'x': [], 'y': []}

        for row in reader:
            if len(row) < 4 + num_jammers * 2:
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
    
    # 予測リストの描画
    if prediction_snapshots:
        for idx, shot in enumerate(prediction_snapshots):
            a_pos = shot["agent_pos"]
            all_jam_preds = shot["preds"]
            
            # ① 予測が行われた位置に小さな黒丸を打つ
            trigger_label = "Prediction Trigger" if idx == 0 else ""
            plt.scatter(a_pos[0], a_pos[1], color='black', marker='o', s=30, zorder=6, label=trigger_label)
            
            # モンテカルロ判定：ジャマー数より予測軌道の数が多い場合は透過度を下げる
            is_monte_carlo = len(all_jam_preds) > num_jammers
            alpha_val = 0.15 if is_monte_carlo else 0.9
            lw_val = 1.0 if is_monte_carlo else 2.5
            
            # ② 予測軌道を描画
            for jam_idx, pred_traj in enumerate(all_jam_preds):
                # どのジャマーに対する予測かを計算して色を合わせる
                # （モンテカルロでジャマー1機につき50本出るような構造に対応）
                target_jammer_id = jam_idx % max(1, num_jammers)
                c = colors[target_jammer_id % len(colors)]
                
                px = [pt[0] for pt in pred_traj]
                py = [pt[1] for pt in pred_traj]
                
                pred_label = "Predicted Traj" if idx == 0 and jam_idx < num_jammers else ""
                plt.plot(px, py, color=c, linestyle=':', alpha=alpha_val, linewidth=lw_val, zorder=4, label=pred_label)

    plt.title(f"Dynamic Jammer Evasion ({now_time})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 凡例の重複を整理して外側に配置
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = {k: v for k, v in by_label.items() if k} # 空のラベルを除去
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    img_path = os.path.join(config.OUTPUT_DIR, f"{now_time}_trajectory.png")
    plt.savefig(img_path) 
    plt.close()
    print(f"軌跡の画像を保存しました: {img_path}")

# CSVデータから時系列のGIFアニメーションを生成する関数
def create_animation_from_csv(now_time):
    csv_path = os.path.join(config.OUTPUT_DIR, f"{now_time}_test_log.csv")

    x_history, y_history = [], []
    jammer_histories = {} 
    
    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        if len(header) < 3:
            raise ValueError(f"Invalid CSV header in {csv_path}")
        num_jammers = (len(header) - 4) // 2

        for i in range(num_jammers):
            jammer_histories[i] = {'x': [], 'y': []}

        for row in reader:
            if len(row) < 4 + num_jammers * 2:
                continue
            x_history.append(float(row[1]))
            y_history.append(float(row[2]))
            for i in range(num_jammers):
                jammer_histories[i]['x'].append(float(row[3 + i*2]))
                jammer_histories[i]['y'].append(float(row[4 + i*2]))

    if not x_history:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')
    gx, gy = config.GOAL_POS
    ax.scatter(gx, gy, color='red', marker='*', s=200, label='Goal', zorder=5)

    # 動かすオブジェクトの初期化
    agent_dot, = ax.plot([], [], 'bo', markersize=8, zorder=6, label='Agent')
    agent_tail, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.4, zorder=3)

    jammer_dots = []
    jammer_tails = []
    jammer_circles = []
    colors = ['orange', 'purple', 'cyan', 'brown', 'pink']

    for i in range(num_jammers):
        c = colors[i % len(colors)]
        jd, = ax.plot([], [], marker='o', color=c, markersize=8, zorder=5, label=f'Jammer {i+1}')
        jt, = ax.plot([], [], color=c, linestyle='--', linewidth=1.5, alpha=0.4, zorder=3)
        jc = patches.Circle((20, 20), radius=config.OBSTACLE_RADIUS, color='grey', alpha=0.4, zorder=2)
        ax.add_patch(jc)
        
        jammer_dots.append(jd)
        jammer_tails.append(jt)
        jammer_circles.append(jc)

    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.set_title(f"Dynamic Jammer Evasion Animation ({now_time})")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 凡例の設定
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1))
    fig.tight_layout()

    # 初期化関数
    def init():
        agent_dot.set_data([], [])
        agent_tail.set_data([], [])
        for i in range(num_jammers):
            jammer_dots[i].set_data([], [])
            jammer_tails[i].set_data([], [])
            jammer_circles[i].center = (20, 20)
        time_text.set_text('')
        return [agent_dot, agent_tail, time_text] + jammer_dots + jammer_tails + jammer_circles

    # コマごとの更新関数
    def update(frame):
        # エージェントの現在位置と軌跡
        agent_dot.set_data([x_history[frame]], [y_history[frame]])
        agent_tail.set_data(x_history[:frame+1], y_history[:frame+1])

        # ジャマーの現在位置と軌跡とバリア円
        for i in range(num_jammers):
            jx = jammer_histories[i]['x'][frame]
            jy = jammer_histories[i]['y'][frame]
            jammer_dots[i].set_data([jx], [jy])
            jammer_tails[i].set_data(jammer_histories[i]['x'][:frame+1], jammer_histories[i]['y'][:frame+1])
            jammer_circles[i].center = (jx, jy)

        time_text.set_text(f'Step: {frame}')
        return [agent_dot, agent_tail, time_text] + jammer_dots + jammer_tails + jammer_circles

    # アニメーション作成 (interval=50 はコマの切り替え速度: 50ミリ秒)
    ani = animation.FuncAnimation(fig, update, frames=len(x_history), init_func=init, blit=True, interval=50)

    gif_path = os.path.join(config.OUTPUT_DIR, f"{now_time}_animation.gif")
    ani.save(gif_path, writer='pillow')
    plt.close()
    print(f"動的軌跡のGIFアニメーションを保存しました: {gif_path}")

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    #---------------wrapper装備----------------------
    # 0. まず生の環境を作成
    raw_env = MyJammerEnv()

    # 【実験1】純粋な強化学習（座標のみ）
    # env = raw_env

    # 【実験2】強化学習 ＋ 直線予測　＋　シールド
    # env = TrajectoryPredictionWrapper(raw_env, history_length=2, horizon_steps=20)
    # env = SafetyShieldWrapper(env, lookahead_steps=15, safety_margin=0.35)

    # 【実験3】強化学習 ＋ 速度ベクトル入力
    # env = VelocityObservationWrapper(raw_env)

    # 【実験4】ハイブリッド（速度入力 ＋ CV予測シールド）
    # env = VelocityObservationWrapper(raw_env)
    # env = TrajectoryPredictionWrapper(env, history_length=2, horizon_steps=20)
    # env = SafetyShieldWrapper(env, lookahead_steps=15, safety_margin=0.35)

    # 【実験5】最新ハイブリッド（速度入力 ＋ カルマンフィルタ予測シールド）
    # env = VelocityObservationWrapper(raw_env)
    # env = KalmanPredictionWrapper(env, horizon_steps=20)
    # env = SafetyShieldWrapper(env, lookahead_steps=15, safety_margin=0.35)

    # 【実験6】カルマン予測 ＋ 人工ポテンシャル法シールド（APF）
    # env = VelocityObservationWrapper(raw_env)
    # env = KalmanPredictionWrapper(env, horizon_steps=20)
    # env = PotentialFieldShieldWrapper(env, lookahead_steps=15, safety_margin=0.35, k_rep=0.05)

    # 実験6のノイズを抑えた
    env = VelocityObservationWrapper(raw_env)
    env = KalmanPredictionWrapper(env, horizon_steps=8)
    env = PotentialFieldShieldWrapper(env, lookahead_steps=8, safety_margin=0.35, k_rep=0.01)

    # 【実験7】モンテカルロ法予測 ＋　人工ポテンシャルシールド（APF）
    # env = VelocityObservationWrapper(raw_env)
    # env = MonteCarloPredictionWrapper(env, horizon_steps=20, num_samples=50)
    # env = PotentialFieldShieldWrapper(env, lookahead_steps=15, safety_margin=0.35, k_rep=0.05)

    # 実験7のパラメータ変更
    # env = VelocityObservationWrapper(raw_env)
    # env = MonteCarloPredictionWrapper(env, horizon_steps=10, num_samples=30)
    # env = PotentialFieldShieldWrapper(env, lookahead_steps=5, safety_margin=0.35, k_rep=0.01)
    #------------------------------------------------

    # 実験開始時刻
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_id = f"{config.EXP_NAME}_{now_time}"

    print(f"\n=========================================")
    print(f" 実験開始: {config.EXP_NAME} (ID: {run_id})")
    print(f"=========================================\n")

    # 環境envを利用して学習を実行する
    model, rewards_history = learn_td3(env, now_time)

    ######## 学習時のスコアの描画
    try:
        draw_score(now_time, rewards_history)
    except Exception as e:
        print(f"draw_score の実行に失敗しました: {e}")

    ######## 移動平均の描画
    try:
        draw_score_moving_average(now_time, rewards_history, window=10, min_periods=1, plot_raw=False)
    except Exception as e:
        print(f"draw_score_moving_average の実行に失敗: {e}")

    ######## actual_test から予測リストを受け取る
    try:
        pred_snapshots = actual_test(now_time, model, env)
    except Exception as e:
        print(f"actual_test の実行に失敗しました: {e}")
        pred_snapshots = None

    ######## 受け取ったリストをそのまま draw_from_csv に引き渡す
    try:
        draw_from_csv(now_time, pred_snapshots)
    except Exception as e:
        print(f"draw_from_csv の実行に失敗しました: {e}")

    ######## 同じCSVを読み込んでGIFアニメーションを出力する
    try:
        create_animation_from_csv(now_time)
    except Exception as e:
        print(f"create_animation_from_csv の実行に失敗しました: {e}")
    
    return

if __name__ == "__main__":
    main()