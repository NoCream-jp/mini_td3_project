import math
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from gymnasium import spaces

@dataclass
class JammerState:
    x: float = 1.0
    y: float = 1.0
    psi: float = 0.0 # 向いている角度
    v: float = 0.05  # 移動速度（動くように0.05を設定！）

class MyJammerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 観測空間を拡張！ [自分のx, 自分のy, ジャマーのx, ジャマーのy] の4つの数値が見えるようになる
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(4,), dtype=np.float32)
        # 行動空間は今まで通り [dx, dy]
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        
        self.location = np.zeros(2, dtype=np.float32)
        self.jam = JammerState()
        
        self.steps_limit_with_learning = 200
        self.current_step = 0
        self.obstacle_radius = 0.5
        self.goal_tol_m = 0.1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # ジャマーを初期位置にリセット（角度はランダムにして、毎回違う方向に動くようにする）
        self.jam = JammerState(
            x=1.0, 
            y=1.0, 
            psi=float(self.np_random.uniform(-np.pi, np.pi)), 
            v=0.05
        )

        # エージェントがジャマーの中に入らないように初期化
        while True:
            self.location = self.np_random.uniform(low=-2.0, high=2.0, size=(2,)).astype(np.float32)
            dist_to_obstacle = np.linalg.norm(self.location - np.array([self.jam.x, self.jam.y]))
            if dist_to_obstacle > self.obstacle_radius:
                break

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. ジャマー（動的障害物）の更新 (先輩のロジックを簡略化して再現)
        # 毎ステップ、少しずつ回転しながら進む（円を描くように動く）
        yaw_rate = 0.1
        self.jam.psi += yaw_rate
        self.jam.x += self.jam.v * math.cos(self.jam.psi)
        self.jam.y += self.jam.v * math.sin(self.jam.psi)
        
        # 2. エージェントの更新
        self.location += action

        # 3. 距離計算と終了判定
        dist_to_goal = np.linalg.norm(self.location)
        jam_pos = np.array([self.jam.x, self.jam.y])
        dist_to_obstacle = np.linalg.norm(self.location - jam_pos)

        finish_flag = False
        over_step_flag = self.steps_limit_with_learning <= self.current_step

        # 4. 報酬計算 (前回修正した特大ボーナス入り)
        if dist_to_obstacle <= self.obstacle_radius:
            reward = -1000.0
            finish_flag = True
        elif dist_to_goal <= self.goal_tol_m:
            reward = 1000.0
            finish_flag = True
        else:
            reward = -float(dist_to_goal)

        return self._get_obs(), reward, finish_flag, over_step_flag, {}

    def _get_obs(self):
        # AIの「目」となる情報を配列にまとめる
        obs = np.array([
            self.location[0], 
            self.location[1], 
            self.jam.x, 
            self.jam.y
        ], dtype=np.float32)
        return obs