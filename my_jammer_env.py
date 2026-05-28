import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config

# "liner_cross"なら直線運動して壁で反射
# そうでなければself.typeによって計算
# 直線の計算とupdate_positionが分離してるのが気持ち悪い
class JammerState:
    def __init__(self, config_dict):
        self.type = config_dict.get("type", "circle")
        self.speed = config_dict.get("speed", 0.05)
        
        # linear_cross用の初期化
        if self.type == "linear_cross":
            # 初期位置を固定（右下 [2.0, -2.0]）
            self.x = 2.0
            self.y = -2.0
            # 進行方向ベクトル：右下(2,-2)から左上(-2,2)へ向かう単位ベクトル
            # 方向は (-1, 1) なので、長さで割って正規化する
            self.dir_x = -1.0 / math.sqrt(2.0)
            self.dir_y = 1.0 / math.sqrt(2.0)
        else:
            # 既存の円・8の字用の初期化
            self.cx = config_dict.get("center", [0.0, 0.0])[0]
            self.cy = config_dict.get("center", [0.0, 0.0])[1]
            self.size = config_dict.get("size", 1.0)
            self.t = math.radians(config_dict.get("angle", 0.0))
            self.x = 0.0
            self.y = 0.0
            self.update_position()
        
    def update(self):
        """毎ステップ座標を更新する"""
        if self.type == "linear_cross":
            # 進行方向へ速度分だけ進む
            next_x = self.x + self.speed * self.dir_x
            next_y = self.y + self.speed * self.dir_y
            
            # 壁（-2.0 または 2.0）に達したら進行方向を反転（往復パトロール）
            if next_x < -2.0 or next_x > 2.0 or next_y < -2.0 or next_y > 2.0:
                self.dir_x = -self.dir_x
                self.dir_y = -self.dir_y
                # 枠外にはみ出さないようにクリップ
                next_x = np.clip(next_x, -2.0, 2.0)
                next_y = np.clip(next_y, -2.0, 2.0)
                
            self.x = next_x
            self.y = next_y
        else:
            # 既存の円・8の字の更新
            self.t += self.speed
            self.update_position()

    def update_position(self):
        """円軌道と8の字軌道の幾何学計算"""
        if self.type == "figure8":
            self.x = self.cx + self.size * math.sin(self.t)
            self.y = self.cy + self.size * math.sin(2.0 * self.t)
        elif self.type == "circle":
            self.x = self.cx + self.size * math.cos(self.t)
            self.y = self.cy + self.size * math.sin(self.t)

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