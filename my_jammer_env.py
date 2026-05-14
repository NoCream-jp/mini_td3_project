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
            if self.obstacle_radius < dist_to_obstacle:
                break

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # ==========================================
        # 1. ジャマーの更新（原点周りの円運動）
        # ==========================================
        # 現在のゴール(0,0)からの距離（軌道半径）を計算
        """
        orbit_radius = math.hypot(self.jam.x, self.jam.y)
        
        if orbit_radius > 1e-5: # 原点にピッタリ重なっている場合のゼロ割りエラー防止
            # 現在の原点から見た角度を計算
            current_angle = math.atan2(self.jam.y, self.jam.x)
            
            # 角速度を計算 (角速度 ω = 速度 v / 半径 r)
            angular_speed = self.jam.v / orbit_radius
            
            # 角度を少し進めて、新しいX, Y座標を再計算
            new_angle = current_angle + angular_speed
            self.jam.x = orbit_radius * math.cos(new_angle)
            self.jam.y = orbit_radius * math.sin(new_angle)
            
            # AIの内部計算用に、進行方向(psi)を円の接線方向に向けておく
            self.jam.psi = new_angle + (math.pi / 2.0)
        """
        # ==========================================
        # 2. エージェントの更新（壁のクリッピング対応済）
        # ==========================================
        next_location = self.location + action
        # -2.0 ～ 2.0 の範囲にクリッピング
        clipped_location = np.clip(next_location, -2.0, 2.0)
        # クリップされたか（見えない壁にぶつかったか）を判定
        hit_wall = not np.array_equal(next_location, clipped_location)

        self.location = clipped_location # クリップされた安全な座標を確定

        # ==========================================
        # 3. 距離計算と終了判定
        # ==========================================
        dist_to_goal = np.linalg.norm(self.location)
        jam_pos = np.array([self.jam.x, self.jam.y])
        dist_to_obstacle = np.linalg.norm(self.location - jam_pos)

        finish_flag = False
        over_step_flag = self.steps_limit_with_learning <= self.current_step

        # ==========================================
        # 4. 報酬計算
        # ==========================================
        if dist_to_obstacle <= self.obstacle_radius:
            reward = config.OBSTACLE_REWARD
            finish_flag = True
        elif dist_to_goal <= self.goal_tol_m:
            reward = config.GOAL_REWARD
            finish_flag = True
        else:
            reward = -float(dist_to_goal)
            if hit_wall:
                reward += config.WALL_PENALTY # 距離ペナルティに加えて、壁ドンに対する罰

        return self._get_obs(), reward, finish_flag, over_step_flag, {}

    def _get_obs(self):
        obs = np.array([
            self.location[0], 
            self.location[1], 
            self.jam.x, 
            self.jam.y
        ], dtype=np.float32)
        return obs