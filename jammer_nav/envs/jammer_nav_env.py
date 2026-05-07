# jammer_nav/envs/jammer_nav_env.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding  # ← 追加：seedingエラー対策

def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

@dataclass
class JammerParams:
    v_mean: float = 0.0
    v_std: float = 0.0
    turn_rate_std: float = 0.0
    radius_m: float = 0.5    # 学習側の obstacle_radius=0.5 に合わせる

    yaw_rate_max: float = np.deg2rad(90.0)
    v_min: float = 0.0
    v_max: float = 1.0
    v_track_tau: float = 0.3

    mode: str = "spin"
    tight_spin: bool = False
    spin_dir: float = +1.0
    motion_basis: str = "x"
    v_min_eps: float = 0.05

@dataclass
class JammerState:
    x: float = 1.0   # 学習側の障害物初期位置(1.0, 1.0)に合わせる
    y: float = 1.0
    psi: float = 0.0
    v: float = 0.0

@dataclass
class DynParams:
    dt: float = 0.1
    goal_tol_m: float = 0.1

class JammerNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        
        # --- 学習側のインターフェースに合わせる ---
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        # 型ヒントを追加して、エディタにBox型であることを認識させる
        self.action_space: spaces.Box = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        
        self.location = np.zeros(2, dtype=np.float32)
        
        self.steps_limit_with_learning = 200
        self.current_step = 0
        
        # --- Jammer（障害物）の設定 ---
        self.jp = JammerParams()
        self.jam = JammerState()
        self.obstacle_radius = self.jp.radius_m
        self.use_jammer = True
        self.dyn = DynParams()

        if seed is not None:
            # 修正：seedingエラー対策のため、インポートしたseedingを使用
            self.np_random, _ = seeding.np_random(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset()
            
        self.current_step = 0

        # Jammerの初期位置
        self.jam = JammerState(x=1.0, y=1.0, psi=0.0, v=0.0)

        # 障害物の中に入らないようにスタート地点を決定
        while True:
            self.location = self.np_random.uniform(low=-2.0, high=2.0, size=(2,)).astype(np.float32)
            dist_to_obstacle = np.linalg.norm(self.location - np.array([self.jam.x, self.jam.y]))
            if dist_to_obstacle > self.obstacle_radius:
                break

        return self.location.copy(), {}

    def step(self, action: np.ndarray):
        self.current_step += 1

        # 修正：low/highエラー対策として、assertを使ってBox型であることを保証する
        assert isinstance(self.action_space, spaces.Box)
        
        # アクションの適用
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.location += action

        # 1) Jammer(動的障害物)の更新
        if self.use_jammer:
            self._update_jammer()

        # 2) 距離計算
        dist_to_goal = float(np.linalg.norm(self.location))
        jam_pos = np.array([self.jam.x, self.jam.y])
        dist_to_obstacle = float(np.linalg.norm(self.location - jam_pos))

        # 3) 終了判定フラグ
        finish_flag = False
        over_step_flag = (self.current_step >= self.steps_limit_with_learning)

        # 4) 報酬・衝突判定
        if dist_to_obstacle <= self.obstacle_radius:
            reward = -1000.0
            finish_flag = True
        elif dist_to_goal <= self.dyn.goal_tol_m:
            reward = -dist_to_goal
            finish_flag = True
        else:
            reward = -dist_to_goal

        return self.location.copy(), reward, finish_flag, over_step_flag, {}

    def _update_jammer(self):
        dt = self.dyn.dt
        self.jam.x += self.jam.v * math.cos(self.jam.psi) * dt
        self.jam.y += self.jam.v * math.sin(self.jam.psi) * dt