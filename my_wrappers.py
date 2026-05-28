import gymnasium as gym
import numpy as np
from collections import deque
from typing import cast
from my_jammer_env import MyJammerEnv

class TrajectoryPredictionWrapper(gym.Wrapper):
    def __init__(self, env, history_length=5, horizon_steps=20):
        super().__init__(env)
        self.history_length = max(2, history_length)
        self.horizon_steps = horizon_steps
        
        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        self.num_jammers = raw_env.num_jammers
        self.jammer_histories = [deque(maxlen=self.history_length) for _ in range(self.num_jammers)]

    def reset(self, seed=None, options=None):
        # まず元の環境をリセット（これでジャマーが config の初期位置 [2.0, -2.0] に配置される）
        obs, info = self.env.reset(seed=seed, options=options)
        
        self.jammer_histories = [deque(maxlen=self.history_length) for _ in range(self.num_jammers)]
        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        
        # ⭕ 修正：確実に [2.0, -2.0] という初期位置を履歴の先頭に叩き込む
        for i, jam in enumerate(raw_env.jammers):
            self.jammer_histories[i].append(np.array([jam.x, jam.y], dtype=np.float32))
            
        info['jam_preds'] = self._predict_trajectories()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        
        for i, jam in enumerate(raw_env.jammers):
            self.jammer_histories[i].append(np.array([jam.x, jam.y], dtype=np.float32))
            
        info['jam_preds'] = self._predict_trajectories()
        
        return obs, reward, terminated, truncated, info

    def _predict_trajectories(self):
        all_preds = []
        
        for hist in self.jammer_histories:
            pred_traj = []
            current_pos = hist[-1]
            sim_x, sim_y = current_pos[0], current_pos[1]
            
            if len(hist) >= 2:
                oldest_pos = hist[0]
                dx = (current_pos[0] - oldest_pos[0]) / (len(hist) - 1)
                dy = (current_pos[1] - oldest_pos[1]) / (len(hist) - 1)
            else:
                # ⭕ 修正：ゲーム開始直後（履歴が1つしかない時）も、停止予測ではなく、
                # configの初期速度の方向（ linear_cross の方向）へ正しくベクトルを伸ばす保険
                raw_env = cast(MyJammerEnv, self.env.unwrapped)
                # 今回は1機のみかつ linear_cross を想定しているため、初期ベクトルを明示
                dx = -0.1 / np.sqrt(2.0)
                dy = 0.1 / np.sqrt(2.0)
                
            sim_dx, sim_dy = dx, dy
            
            for _ in range(self.horizon_steps):
                next_x = sim_x + sim_dx
                next_y = sim_y + sim_dy
                
                if next_x < -2.0 or next_x > 2.0:
                    sim_dx = -sim_dx
                    next_x = np.clip(next_x, -2.0, 2.0)
                if next_y < -2.0 or next_y > 2.0:
                    sim_dy = -sim_dy
                    next_y = np.clip(next_y, -2.0, 2.0)
                    
                sim_x, sim_y = next_x, next_y
                pred_traj.append((sim_x, sim_y))
                
            all_preds.append(pred_traj)
            
        return all_preds