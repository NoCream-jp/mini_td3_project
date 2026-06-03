import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config

class JammerState:
    def __init__(self, config_dict):
        self.type = config_dict.get("type", "circle")
        self.speed = config_dict.get("speed", 0.05)
        
        # 内部時間・位相としてのパラメータ（全軌道共通）
        self.t = math.radians(config_dict.get("angle", 0.0))
        
        # サイン波（sin_wave）および直線（linear_cross）用の幾何学設定
        if self.type in ["sin_wave", "linear_cross"]:
            # 指定された始点と終点（デフォルトは右下から左上への対角線）
            self.start_pos = np.array(config_dict.get("start_pos", [2.0, -2.0]), dtype=np.float32)
            self.end_pos = np.array(config_dict.get("end_pos", [-2.0, 2.0]), dtype=np.float32)
            
            # 波のカスタムパラメータ
            self.amplitude = config_dict.get("amplitude", 0.5)
            self.frequency = config_dict.get("frequency", 2.0)
            
            # 往復運動を管理するための内部進行度 (0.0 から 1.0 の間を往復)
            self.progress = 0.0
            self.forward = True # True: 往路(始->終), False: 復路(終->始)
            
            # 直線移動ベクトルの算出（向きを揃えるため）
            direction = self.end_pos - self.start_pos
            self.total_dist = np.linalg.norm(direction)
            self.dir_unit = direction / (self.total_dist + 1e-6)
            
            # 垂直ベクトルの算出（波を進行方向に対して横に揺らすため）
            self.perpendicular_unit = np.array([-self.dir_unit[1], self.dir_unit[0]], dtype=np.float32)
        else:
            # 既存の円・8の字用の幾何学設定
            self.cx = config_dict.get("center", [0.0, 0.0])[0]
            self.cy = config_dict.get("center", [0.0, 0.0])[1]
            self.size = config_dict.get("size", 1.0)
            
        self.x = 0.0
        self.y = 0.0
        self.update_position()
        
    def update(self):
        """毎ステップ時間を進めて座標を自動更新する（統一窓口）"""
        if self.type in ["sin_wave", "linear_cross"]:
            # 1ステップあたりの進捗度（割合）を計算
            step_progress = self.speed / (self.total_dist + 1e-6)
            
            if self.forward:
                self.progress += step_progress
                if self.progress >= 1.0:
                    self.progress = 1.0
                    self.forward = False # 終点に達したら反転
            else:
                self.progress -= step_progress
                if self.progress <= 0.0:
                    self.progress = 0.0
                    self.forward = True # 始点に達したら反転
        else:
            # 円・8の字は角度時間を進める
            self.t += self.speed
            
        self.update_position()

    def update_position(self):
        """全ての幾何学計算をここに集約"""
        if self.type == "sin_wave":
            # 1. 基準線（直線ルート）上の現在位置を算出
            base_pos = self.start_pos + (self.end_pos - self.start_pos) * self.progress
            # 2. 進捗に応じたサイン波の横揺れ（変位）を計算
            # 往復で滑らかに繋がるよう、progressに π * frequency を掛ける
            wave_offset = self.amplitude * math.sin(self.progress * math.pi * self.frequency)
            # 3. 基準線に垂直な波を合成
            final_pos = base_pos + self.perpendicular_unit * wave_offset
            self.x, self.y = final_pos[0], final_pos[1]
            
        elif self.type == "linear_cross":
            # 揺れのない純粋な往復直線運動
            final_pos = self.start_pos + (self.end_pos - self.start_pos) * self.progress
            self.x, self.y = final_pos[0], final_pos[1]
            
        elif self.type == "figure8":
            self.x = self.cx + self.size * math.sin(self.t)
            self.y = self.cy + self.size * math.sin(2.0 * self.t)
            
        elif self.type == "circle":
            self.x = self.cx + self.size * math.cos(self.t)
            self.y = self.cy + self.size * math.sin(self.t)
            
        # 共通の壁クリップ処理（予測や描画の枠外はみ出し防止）
        self.x = np.clip(self.x, -2.0, 2.0)
        self.y = np.clip(self.y, -2.0, 2.0)

class MyJammerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_jammers = len(config.JAMMER_CONFIGS)
        
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

        # options にスタート位置の指定があればそれを使う、無ければランダムにする
        if options is not None and "start_pos" in options:
            self.location = np.array(options["start_pos"], dtype=np.float32)
        else:
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
                raise RuntimeError("Could not sample a start position...")

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