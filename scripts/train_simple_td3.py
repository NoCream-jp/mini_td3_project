import datetime
import os
import csv
import numpy as np
from numpy.random import f
# 環境
import gymnasium as gym
# 学習
from stable_baselines3 import TD3
# 描画
import matplotlib.pyplot as plt
# 障害物描画
import matplotlib.patches as patches

# 出力用ディレクトリ
OUTPUT_DIR = "outputs"

# 学習時のステップ数上限
# TOTAL_TIMESTEPS = 1000
TOTAL_TIMESTEPS = 100_000

"""
今回は障害物をかわしてゴールを目指す
次回は障害物を動かしてみる？
"""

# 環境設定クラス
# 学習時の挙動を定義．大事な追加分は特にsteps_limit_with_learning
class RandomStartEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 観察エリア定義：-2 <= x <= 2, -2 <= y <= 2
        self.observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        # 動けるエリア定義：-0.1 <= x <= 0.1, -0.1 <= y <= 0.1
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self.location = np.zeros(2, dtype=np.float32)
        ## 1エピソードで歩めるstep数に制限をかける．見当違いで全ステップ数を消費しないように
        self.steps_limit_with_learning = 200
        self.current_step = 0

        # 障害物エリア定義
        self.obstacle_center = np.array([1.0, 1.0], dtype=np.float32)
        self.obstacle_radius = 0.5

    # エピソードに一回，初期化時に呼ばれる．
    # スタート地点を決めなおす．
    def reset(self, seed=None, options=None):
        # seedは同じランダム数列を再現するためのキー．
        # これがないと「まったく同じランダムな試練」を異なる方策同士で共有できない．
        # たくさんある乱数表のうち{seed}番目を使えという指示のようなもの．
        super().reset(seed=seed)
        # スタート地点設定
        self.location = self.np_random.uniform(low=-2.0, high=2.0, size=(2,)).astype(np.float32)
        # エピソードごとにstep数をリセット
        self.current_step = 0

        # 障害物の中に入らないようにする
        while True:
            self.location = self.np_random.uniform(low=-2.0, high=2.0, size=(2,)).astype(np.float32)
            dist_to_obstacle = np.linalg.norm(self.location - self.obstacle_center)
            if self.obstacle_radius < dist_to_obstacle:
                break # 障害物の外なら確定してループを抜ける

        return self.location, {}

    # 一歩
    def step(self, action):
        # step数制限までは歩き続ける．
        self.current_step += 1
        # 移動する．np.arrayだから+=で記述できる.
        self.location += action

        # ゴールへの距離計算
        dist_to_goal = np.linalg.norm(self.location)
        # 障害物への距離計算
        dist_to_obstacle = np.linalg.norm(self.location - self.obstacle_center)

        # reward = -dist_to_goal #
        # 終了条件フラグ
        finish_flag = False
        # リミット超過しなかったかのフラグ
        over_step_flag = self.steps_limit_with_learning <= self.current_step

        if dist_to_obstacle <= self.obstacle_radius:
            # 障害物に衝突した場合強制終了
            reward = -1000
            finish_flag = True
        elif dist_to_goal <= 0.1:
            # ゴールに到達した場合強制終了
            reward = -float(dist_to_goal)
            finish_flag = True
        else:
            # 通常移動
            reward = -float(dist_to_goal)

        return self.location, reward, finish_flag, over_step_flag, {} # 状態，報酬，終了したか否か，あとは便宜上

# 学習(td3)
def learn_td3(env):
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(OUTPUT_DIR, "simple_td3_model"))
    return model

def save_result(now_time, model, env):
    with open(os.path.join(OUTPUT_DIR, f"test_{now_time}_log.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "x", "y"])
        location, _ = env.reset()
        # 前回のlocationを使ってステップを1進める．
        # ノイズ一切なし（deterministic = True）= 決定論的
        for i in range(1000):
            # 予測して
            action, _ = model.predict(location, deterministic=True)
            # それをもとに1ステップ進める
            location, _, finish_flag, over_step_flag, _ = env.step(action)
            writer.writerow([i, location[0], location[1]]) # 行番号，x, yをcsvに記録
            # 学習が完了（終了条件: 原点に距離1以内で接近）していれば終わり
            if finish_flag or over_step_flag:
                break

def draw_from_csv(now_time):
    """
    指定された日時のCSVファイルから軌跡を読み込み、
    matplotlibで描画して画像として保存する
    """
    # 現在時刻からcsvの名前を作ったので，それを利用して持ってこれる
    csv_path = os.path.join(OUTPUT_DIR, f"test_{now_time}_log.csv")

    x_history = []
    y_history = []
    # CSVの読み込み
    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            x_history.append(float(row[1]))
            y_history.append(float(row[2]))

    # --- 描画設定 ---
    plt.figure(figsize=(6, 6)) # 正方形のグラフにする
    # 観察エリアの制限 (-2.0 から 2.0)
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)

    # ゴール地点（原点）をプロット
    plt.scatter(0, 0, color='red', marker='*', s=200, label='Goal (0,0)')

    # 障害物の描画
    obstacle_circle = patches.Circle((1.0, 1.0), radius=0.5, color='grey', alpha=0.5, label='Obstacle')
    plt.gca().add_patch(obstacle_circle)

    # 軌跡を折れ線グラフでプロット
    plt.plot(x_history, y_history, color='blue', marker='.', linestyle='-', linewidth=1.5, label='Trajectory')
    # スタート地点を緑色の点で強調
    plt.scatter(x_history[0], y_history[0], color='green', marker='o', s=100, label='Start')
    # 見た目の調整
    plt.title(f"Agent Trajectory ({now_time})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend() # 凡例を表示
    # 画像として保存して閉じる
    img_path = os.path.join(OUTPUT_DIR, f"trajectory_{now_time}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"軌跡の画像を保存しました: {img_path}")

def main():

    """
    出力用のディレクトリ作成
    環境の初期化
    学習
    テスト走行を1回分記録
    """

    # 出力用のディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 環境を初期化して準備
    env = RandomStartEnv()

    # 学習
    model = learn_td3(env)

    # テスト走行記録
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_result(now_time, model, env)

    # 描画してpngとして保存
    draw_from_csv(now_time)
    return

if __name__ == "__main__":
    main()
