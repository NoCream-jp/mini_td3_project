import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config

import math
import numpy as np

class JammerState:
    def __init__(self, config_dict):
        self.type = config_dict.get("type", "circle")
        self.speed = config_dict.get("speed", 0.05)
        self.noise_std = config_dict.get("noise_std", 0.0) 
        
        # ==========================================
        # ① 直線運動 (linear_cross) の初期化
        # ==========================================
        if self.type == "linear_cross":
            self.start_pos = np.array(config_dict.get("start_pos", [2.0, -2.0]), dtype=np.float32)
            self.end_pos = np.array(config_dict.get("end_pos", [-2.0, 2.0]), dtype=np.float32)
            
            direction = self.end_pos - self.start_pos
            self.total_dist = np.linalg.norm(direction) + 1e-6
            
            self.progress = 0.0
            self.forward = True

        # ==========================================
        # ② サイン波 (sin_wave) の初期化
        # ==========================================
        elif self.type == "sin_wave":
            self.start_pos = np.array(config_dict.get("start_pos", [2.0, -2.0]), dtype=np.float32)
            self.end_pos = np.array(config_dict.get("end_pos", [-2.0, 2.0]), dtype=np.float32)
            self.amplitude = config_dict.get("amplitude", 0.5)
            self.frequency = config_dict.get("frequency", 2.0)
            
            direction = self.end_pos - self.start_pos
            self.total_dist = np.linalg.norm(direction) + 1e-6
            
            dir_unit = direction / self.total_dist
            self.perpendicular_unit = np.array([-dir_unit[1], dir_unit[0]], dtype=np.float32)
            
            self.progress = 0.0
            self.forward = True

        # ==========================================
        # ③ 円・8の字 (circle / figure8) の初期化
        # ==========================================
        elif self.type in ["circle", "figure8"]:
            self.cx = config_dict.get("center", [0.0, 0.0])[0]
            self.cy = config_dict.get("center", [0.0, 0.0])[1]
            self.size = config_dict.get("size", 1.0)
            self.t = math.radians(config_dict.get("angle", 0.0))
        
        # ==========================================
        # ④ 盤面を左右に往復走査の初期化
        # ==========================================
        elif self.type == "serpentine":
            self.margin = float(config_dict.get("margin", 0.2))
            self.lane_spacing = float(config_dict.get("lane_spacing", 0.6))
            if not 0 <= self.margin < 2.0:
                raise ValueError("serpentine: margin は 0 以上 2 未満にしてください")
            if self.lane_spacing <= 0:
                raise ValueError("serpentine: lane_spacing は正の値にしてください")

            self.scan_min = -2.0 + self.margin
            self.scan_max = 2.0 - self.margin
            self.base_pos = np.array(
                [self.scan_min, self.scan_min], dtype=np.float64
            )
            self.scan_phase = "horizontal"
            self.scan_x_dir = 1
            self.scan_y_dir = 1
            self.scan_target = np.array(
                [self.scan_max, self.scan_min], dtype=np.float64
            )
        
        # ==========================================
        # ⑤ 追尾の初期化
        # ==========================================
        elif self.type == "chase":
            self.base_pos = np.clip(
                np.array(
                    config_dict.get("start_pos", [2.0, -2.0]),
                    dtype=np.float64
                ),
                -2.0, 2.0
            )

        self.x = 0.0
        self.y = 0.0
        self.update_position()

    def _next_serpentine_target(self):
        """端に到達したら次の走査目標を設定する。"""
        if self.scan_phase == "horizontal":
            self.scan_phase = "vertical"
            next_y = self.base_pos[1] + self.scan_y_dir * self.lane_spacing

            if next_y >= self.scan_max:
                next_y = self.scan_max
                self.scan_y_dir = -1
            elif next_y <= self.scan_min:
                next_y = self.scan_min
                self.scan_y_dir = 1

            self.scan_target = np.array([self.base_pos[0], next_y])
        else:
            self.scan_phase = "horizontal"
            self.scan_x_dir *= -1
            next_x = self.scan_max if self.scan_x_dir == 1 else self.scan_min
            self.scan_target = np.array([next_x, self.base_pos[1]])
    
    def _advance_serpentine(self):
        """折り返し地点をまたいでも移動距離を一定に保つ。"""
        remaining = self.speed

        while remaining > 1e-10:
            vector = self.scan_target - self.base_pos
            distance = float(np.linalg.norm(vector))

            if distance <= 1e-10:
                self._next_serpentine_target()
                continue

            step = min(remaining, distance)
            self.base_pos += vector / distance * step
            remaining -= step

            if step >= distance - 1e-10:
                self.base_pos = self.scan_target.copy()
                self._next_serpentine_target()

        
    def update(self, target_pos=None):
        """毎ステップ、各軌道に合わせた「時間・進捗」を進める"""
        # 対角線またはsin波なら次を計算して追加するだけ
        if self.type in ["linear_cross", "sin_wave"]:
            step_progress = self.speed / self.total_dist
            
            if self.forward:
                self.progress += step_progress
                if self.progress >= 1.0:
                    self.progress = 1.0
                    self.forward = False
            else:
                self.progress -= step_progress
                if self.progress <= 0.0:
                    self.progress = 0.0
                    self.forward = True
        # 円や八の字なら初期化に従ってステップするだけ            
        elif self.type in ["circle", "figure8"]:
            self.t += self.speed
        
        elif self.type == "serpentine":
           self._advance_serpentine()

        elif self.type == "chase":
            if target_pos is None:
                raise ValueError("chase ジャマーの更新には target_pos が必要です")

            direction = np.asarray(target_pos, dtype=np.float64) - self.base_pos
            distance = float(np.linalg.norm(direction))

            if 1e-10 < distance:
                self.base_pos += direction / distance * min(self.speed, distance)
            
        self.update_position()

    def update_position(self):
        """進んだ「時間・進捗」を元に、実際の X, Y 座標を計算する"""
        if self.type == "linear_cross":
            pos = self.start_pos + (self.end_pos - self.start_pos) * self.progress
            self.x, self.y = pos[0], pos[1]
        
        elif self.type == "sin_wave":
            base_pos = self.start_pos + (self.end_pos - self.start_pos) * self.progress
            wave_offset = self.amplitude * math.sin(self.progress * math.pi * self.frequency)
            pos = base_pos + self.perpendicular_unit * wave_offset
            self.x, self.y = pos[0], pos[1]
        
        elif self.type == "circle":
            self.x = self.cx + self.size * math.cos(self.t)
            self.y = self.cy + self.size * math.sin(self.t)
            
        elif self.type == "figure8":
            self.x = self.cx + self.size * math.sin(self.t)
            self.y = self.cy + self.size * math.sin(2.0 * self.t)
        
        elif self.type in ("serpentine", "chase"):
            self.x, self.y = self.base_pos
            
        # ==========================================
        # ガウスノイズの重畳
        # ==========================================
        if self.noise_std > 0.0:
            self.x += np.random.normal(0, self.noise_std)
            self.y += np.random.normal(0, self.noise_std)

        # 画面外に出ないようにクリップ（全軌道共通）
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
        
        next_location = self.location + action
        clipped_location = np.clip(next_location, -2.0, 2.0)
        hit_wall = not np.array_equal(next_location, clipped_location)
        self.location = clipped_location 

        # すべてのジャマーについて、位置を更新する
        # UAVが移動してから、target_posとしてジャマーに渡す
        for jam in self.jammers:
            jam.update(target_pos=self.location)

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
            reward = -float(dist_to_goal) * 2
            if hit_wall:
                reward += config.WALL_PENALTY

        return self._get_obs(), reward, finish_flag, over_step_flag, {}

    def _get_obs(self):
        obs_list = [self.location[0], self.location[1]]
        for jam in self.jammers:
            obs_list.extend([jam.x, jam.y])
        return np.array(obs_list, dtype=np.float32)