import gymnasium as gym
import numpy as np
from collections import deque

from typing import cast
from my_jammer_env import MyJammerEnv

class TrajectoryPredictionWrapper(gym.Wrapper):
    """
    直近数ステップの座標履歴から「平均差分ベクトル（等速直線）」を計算し、
    未来の予測軌跡を info['jam_preds'] に格納するラッパー。
    """
    def __init__(self, env, history_length=5, horizon_steps=20):
        super().__init__(env)
        self.history_length = max(2, history_length) # 最低2ステップないと差分が取れない
        self.horizon_steps = horizon_steps
        
        # ジャマーの数だけ、履歴保存用のデック（キュー）を用意する
        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        self.num_jammers = raw_env.num_jammers
        self.jammer_histories = [deque(maxlen=self.history_length) for _ in range(self.num_jammers)]

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # リセット時に履歴をクリアし、初期位置を記録
        self.jammer_histories = [deque(maxlen=self.history_length) for _ in range(self.num_jammers)]
        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        
        for i, jam in enumerate(raw_env.jammers):
            self.jammer_histories[i].append(np.array([jam.x, jam.y]))
            
        # 初期状態からの予測（動いていないので停止予測）を info に格納
        info['jam_preds'] = self._predict_trajectories()
        return obs, info

    def step(self, action):
        # 1. 環境を1ステップ進める
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        
        # 2. 最新の座標を履歴に追加（上限を超えると古いものから自動で消える）
        for i, jam in enumerate(raw_env.jammers):
            self.jammer_histories[i].append(np.array([jam.x, jam.y]))
            
        # 3. 予測軌道を計算して info に格納
        info['jam_preds'] = self._predict_trajectories()
        
        return obs, reward, terminated, truncated, info

    def _predict_trajectories(self):
        """履歴から予測軌跡リストを生成する内部関数"""
        all_preds = []
        
        for hist in self.jammer_histories:
            pred_traj = []
            
            # 最新の座標
            current_pos = hist[-1]
            sim_x, sim_y = current_pos[0], current_pos[1]
            
            # 履歴が2つ以上あれば、平均差分ベクトル（速度）を計算
            if len(hist) >= 2:
                oldest_pos = hist[0]
                # (最新 - 最古) / ステップ間隔 = 1ステップあたりの平均移動量
                dx = (current_pos[0] - oldest_pos[0]) / (len(hist) - 1)
                dy = (current_pos[1] - oldest_pos[1]) / (len(hist) - 1)
            else:
                dx, dy = 0.0, 0.0
                
            sim_dx, sim_dy = dx, dy
            
            # 未来へ向けてシミュレーション（horizon_steps分）
            for _ in range(self.horizon_steps):
                next_x = sim_x + sim_dx
                next_y = sim_y + sim_dy
                
                # 壁での反射ロジック（予測が枠外に飛び出さないための補正）
                if next_x < -2.0 or next_x > 2.0:
                    sim_dx = -sim_dx # Xの進行方向を反転
                    next_x = np.clip(next_x, -2.0, 2.0)
                if next_y < -2.0 or next_y > 2.0:
                    sim_dy = -sim_dy # Yの進行方向を反転
                    next_y = np.clip(next_y, -2.0, 2.0)
                    
                sim_x, sim_y = next_x, next_y
                pred_traj.append((sim_x, sim_y))
                
            all_preds.append(pred_traj)
            
        return all_preds