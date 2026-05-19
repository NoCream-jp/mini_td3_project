import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces


# config.py インポート
import config

# Jammer一つのステータス
class JammerState:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.psi = 0.0 # 向きの初期値 reset()のほうでインスタンス化のあとに上書きすれば当然書き変わる。
        self.v = speed
        
    def update(self):
        """
        ジャマーの軌道：向いている方向(psi)へ直進し、壁(-2.0 ~ 2.0)で反射する
        """
        # 1. 向いている方向(psi)へ、スピード(v)の分だけ座標を進める
        next_x = self.x + self.v * math.cos(self.psi)
        next_y = self.y + self.v * math.sin(self.psi)
        
        # 2. X軸の壁（左右の壁）にぶつかったら反射
        if next_x < -2.0 or next_x > 2.0:
            self.psi = math.pi - self.psi  # 左右の反射（角度を反転）
            next_x = np.clip(next_x, -2.0, 2.0)
            
        # 3. Y軸の壁（上下の壁）にぶつかったら反射
        if next_y < -2.0 or next_y > 2.0:
            self.psi = -self.psi           # 上下の反射（角度を反転）
            next_y = np.clip(next_y, -2.0, 2.0)
            
        # 4. 新しい座標を確定
        self.x = next_x
        self.y = next_y

# 環境定義
class MyJammerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # configで設定したジャマーの数を取得
        self.num_jammers = len(config.JAMMER_CONFIGS)
        
        # 観測空間の拡張: 自分のx,y(2) + (ジャマーの数 × x,y(2))
        obs_dim = 2 + (self.num_jammers * 2)
        # 観測空間の定義
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)
        ## actionの空間の制限定義
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        
        self.location = np.zeros(2, dtype=np.float32)
        self.jammers = [] # 複数のジャマーを入れるリスト

        self.steps_limit_with_learning = config.MAX_STEPS_PER_EPISODE
        self.current_step = 0
        self.obstacle_radius = config.OBSTACLE_RADIUS
        self.goal_tol_m = config.GOAL_TOLERANCE

    # リセット関数
    f"""
    エピソード毎に呼ばれる関数で、
    - config内のデータでjammerを初期化する
    - location(エージェントの座標)とjammerの座標とが被っていないかチェックする
    をする。戻り値は
    _get_obs() -> [(エージェントの座標), (jammer_1の座標), (jammer_2の座標), ...] と 仕様上の空集合

    """
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # すべてのジャマーを初期化してリストに格納
        self.jammers = []
        for j_conf in config.JAMMER_CONFIGS:
            jam = JammerState(x=j_conf["pos"][0], y=j_conf["pos"][1], speed=j_conf["speed"])
            jam.psi = math.radians(j_conf.get("angle", 0.0))
            self.jammers.append(jam)

        max_spawn_attempts = 10_000
        for _ in range(max_spawn_attempts):
            self.location = self.np_random.uniform(low=-2.0, high=2.0, size=(2,)).astype(np.float32)
            # すべてのジャマーとの距離を測り、どれか1つでも被っていたらリトライ
            collision = False
            for jam in self.jammers:
                dist = np.linalg.norm(self.location - np.array([jam.x, jam.y]))
                if dist <= self.obstacle_radius:
                    collision = True
                    break
            if not collision:
                break
        else:
            raise RuntimeError(
                "Could not sample a start position that avoids all jammers. "
                "Reduce OBSTACLE_RADIUS or adjust JAMMER_CONFIGS."
            )

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. すべてのジャマーを1歩進める
        for jam in self.jammers:
            jam.update()
        
        # 2. エージェントの更新
        next_location = self.location + action
        clipped_location = np.clip(next_location, -2.0, 2.0)
        hit_wall = not np.array_equal(next_location, clipped_location)
        self.location = clipped_location 

        # 3. 距離計算と終了判定
        dist_to_goal = np.linalg.norm(self.location)

        # 一番近いジャマーとの距離を計算
        min_dist_to_obstacle = float('inf')
        for jam in self.jammers:
            dist = np.linalg.norm(self.location - np.array([jam.x, jam.y]))
            if dist < min_dist_to_obstacle:
                min_dist_to_obstacle = dist

        finish_flag = False
        over_step_flag = self.steps_limit_with_learning <= self.current_step

        # 4. 報酬計算
        if min_dist_to_obstacle <= self.obstacle_radius:
            reward = config.OBSTACLE_REWARD
            finish_flag = True
        elif dist_to_goal <= self.goal_tol_m:
            reward = config.GOAL_REWARD
            finish_flag = True
        else:
            reward = -float(dist_to_goal)
            if hit_wall:
                reward += config.WALL_PENALTY

        return self._get_obs(), reward, finish_flag, over_step_flag, {}

    def _get_obs(self):
        # 自分の座標を入れた後、すべてのジャマーの座標を配列に追加
        obs_list = [self.location[0], self.location[1]]
        for jam in self.jammers:
            obs_list.extend([jam.x, jam.y])
        return np.array(obs_list, dtype=np.float32)