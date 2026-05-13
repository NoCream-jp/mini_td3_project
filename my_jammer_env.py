# my_jammer_env.py
import math
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config  # 変数ファイルをインポート

@dataclass
class JammerState:
    x: float = config.JAMMER_START_POS[0]
    y: float = config.JAMMER_START_POS[1]
    psi: float = 0.0 
    v: float = config.JAMMER_SPEED

class MyJammerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # [自分のx, 自分のy, ジャマーのx, ジャマーのy]
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(4,), dtype=np.float32)
        # [dx, dy]
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        
        self.location = np.zeros(2, dtype=np.float32)
        self.jam = JammerState()
        
        self.steps_limit_with_learning = config.MAX_STEPS_PER_EPISODE
        self.current_step = 0
        self.obstacle_radius = config.OBSTACLE_RADIUS
        self.goal_tol_m = config.GOAL_TOLERANCE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.jam = JammerState(
            x=config.JAMMER_START_POS[0], 
            y=config.JAMMER_START_POS[1], 
            psi=float(self.np_random.uniform(-np.pi, np.pi)), 
            v=config.JAMMER_SPEED
        )

        while True:
            self.location = self.np_random.uniform(low=-2.0, high=2.0, size=(2,)).astype(np.float32)
            dist_to_obstacle = np.linalg.norm(self.location - np.array([self.jam.x, self.jam.y]))
            if dist_to_obstacle > self.obstacle_radius:
                break

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # ジャマーの更新
        yaw_rate = 0.1
        self.jam.psi += yaw_rate
        self.jam.x += self.jam.v * math.cos(self.jam.psi)
        self.jam.y += self.jam.v * math.sin(self.jam.psi)
        
        # エージェントの更新
        self.location += action

        # 距離計算
        dist_to_goal = np.linalg.norm(self.location)
        jam_pos = np.array([self.jam.x, self.jam.y])
        dist_to_obstacle = np.linalg.norm(self.location - jam_pos)

        finish_flag = False
        over_step_flag = self.steps_limit_with_learning <= self.current_step

        # 報酬計算
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
        obs = np.array([
            self.location[0], 
            self.location[1], 
            self.jam.x, 
            self.jam.y
        ], dtype=np.float32)
        return obs