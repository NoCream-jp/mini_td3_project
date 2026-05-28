import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config

class JammerState:
    def __init__(self, config_dict):
        # configの辞書をそのまま受け取って設定を展開する
        self.type = config_dict.get("type", "circle")
        self.cx = config_dict.get("center", [0.0, 0.0])[0]
        self.cy = config_dict.get("center", [0.0, 0.0])[1]
        self.size = config_dict.get("size", 1.0)
        self.speed = config_dict.get("speed", 0.05)
        
        # 内部時計（時間 t）として扱う。angleでスタート位置をずらせる。
        self.t = math.radians(config_dict.get("angle", 0.0))
        self.update_position()
        
    def update(self):
        """毎ステップ時間を進めて座標を再計算する"""
        self.t += self.speed
        self.update_position()

    def update_position(self):
        """時間に依存して幾何学的な軌道を描く"""
        if self.type == "figure8":
            # 8の字軌道（リサージュ図形）。対角線を何度も塞ぐ意地悪な軌道。
            self.x = self.cx + self.size * math.sin(self.t)
            self.y = self.cy + self.size * math.sin(2.0 * self.t)
        elif self.type == "circle":
            # 単純な円軌道
            self.x = self.cx + self.size * math.cos(self.t)
            self.y = self.cy + self.size * math.sin(self.t)
        else:
            # 指定がない場合は中心で停止
            self.x = self.cx
            self.y = self.cy

class MyJammerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_jammers = len(config.JAMMER_CONFIGS)
        
        # 
        obs_dim = 2 + (self.num_jammers * 2)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        
        self.location = np.zeros(2, dtype=np.float32)
        self.jammers = []

        self.steps_limit_with_learning = config.MAX_STEPS_PER_EPISODE
        # ステップ数
        self.current_step = 0
        self.obstacle_radius = config.OBSTACLE_RADIUS
        self.goal_tol_m = config.GOAL_TOLERANCE
        
        # ゴール座標をconfigからNumpy配列として保持しておく
        self.goal_pos = np.array(config.GOAL_POS, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.jammers = []
        for j_conf in config.JAMMER_CONFIGS:
            jam = JammerState(j_conf)
            self.jammers.append(jam)

        # 不都合な初期位置は10000回までリセットし続ける
        max_spawn_attempts = 10_000
        for _ in range(max_spawn_attempts):
            self.location = self.np_random.uniform(low=-2.0, high=2.0, size=(2,)).astype(np.float32)
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
        
        for jam in self.jammers:
            jam.update()
        
        next_location = self.location + action
        clipped_location = np.clip(next_location, -2.0, 2.0)
        hit_wall = not np.array_equal(next_location, clipped_location)
        self.location = clipped_location 

        # goalへの距離
        dist_to_goal = np.linalg.norm(self.location - self.goal_pos)

        min_dist_to_obstacle = float('inf')
        for jam in self.jammers:
            dist = np.linalg.norm(self.location - np.array([jam.x, jam.y]))
            if dist < min_dist_to_obstacle:
                min_dist_to_obstacle = dist

        finish_flag = False
        over_step_flag = self.steps_limit_with_learning <= self.current_step

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
        obs_list = [self.location[0], self.location[1]]
        for jam in self.jammers:
            obs_list.extend([jam.x, jam.y])
        return np.array(obs_list, dtype=np.float32)