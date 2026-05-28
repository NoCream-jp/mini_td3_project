import math
import gymnasium as gym
import numpy as np
from collections import deque
from typing import cast

import config
from my_jammer_env import MyJammerEnv

# =====================================================================
# 1. 軌道予測ラッパー (The Oracle)
# =====================================================================
class TrajectoryPredictionWrapper(gym.Wrapper):
    """
    直近数ステップの座標履歴から「平均差分ベクトル（等速直線）」を計算し、
    未来の予測軌跡を info['jam_preds'] に格納するラッパー。
    """
    def __init__(self, env, history_length=5, horizon_steps=20):
        super().__init__(env)
        self.history_length = max(2, history_length)
        self.horizon_steps = horizon_steps
        
        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        self.num_jammers = raw_env.num_jammers
        self.jammer_histories = [deque(maxlen=self.history_length) for _ in range(self.num_jammers)]

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        self.jammer_histories = [deque(maxlen=self.history_length) for _ in range(self.num_jammers)]
        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        
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
                # ⭕ 修正：特定の軌道に依存せず、開始直後（履歴不足時）は「停止」として扱う
                # これにより、どんなジャマーを設定してもエラーやズレが起きなくなります
                dx, dy = 0.0, 0.0
                
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


# =====================================================================
# 2. 安全シールドラッパー (The Shield)
# =====================================================================
class SafetyShieldWrapper(gym.Wrapper):
    """
    軌道予測データ(jam_preds)を読み取り、AIが提案した行動(action)を続けた場合に
    未来で衝突しそうなら、行動を安全な方向に強制的に書き換えるラッパー。
    """
    def __init__(self, env, lookahead_steps=15, safety_margin=0.35):
        super().__init__(env)
        self.lookahead_steps = lookahead_steps
        self.safety_margin = safety_margin # 判定半径(0.2) + 余裕(0.15)
        self.latest_preds = []

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # 環境が初期化された時の予測データを取得
        self.latest_preds = info.get('jam_preds', [])
        return obs, info

    def step(self, action):
        # 1. 提案された行動が安全かチェックし、危険なら別の行動に書き換える（リランク）
        safe_action = self._rerank_action(action)
        
        # 2. 「書き換えた安全な行動」で環境を1ステップ進める
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        
        # 3. 次のステップの判断のために、最新の予測データを更新
        self.latest_preds = info.get('jam_preds', [])
        
        # 4. もしシールドが発動（行動が書き換えられた）場合、AIにペナルティを与える
        if not np.array_equal(action, safe_action):
            reward = float(reward) - 5.0 # 「助けてもらうような危険な行動をするな」という教育的指導
            info['shield_activated'] = True
        else:
            info['shield_activated'] = False
            
        return obs, reward, terminated, truncated, info

    def _rerank_action(self, action):
        """行動候補の中から、最も安全で、かつ元の意図に近いものを選ぶ"""
        if not self.latest_preds:
            return action # 予測データが無ければ何もしない
            
        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        current_pos = raw_env.location
        
        # ① まず、AIの本来の行動がそのまま安全かチェック
        if self._is_safe(current_pos, action):
            return action
            
        # ② 危険な場合、少し角度をずらした「回避行動候補」を生成
        candidates = self._generate_candidates(action)
        
        # ③ 候補の中で、安全なものが見つかればそれを採用（少しだけ迂回する）
        for cand in candidates:
            if self._is_safe(current_pos, cand):
                return cand
                
        # ④ どの方向に逃げてもぶつかる絶体絶命の場合、とりあえずブレーキ（急停止）
        return np.zeros(2, dtype=np.float32)

    def _is_safe(self, current_pos, action):
        """指定した行動で進み続けた場合、未来で衝突しないかを判定する"""
        # 予測されている未来の歩数と、見たい歩数の短い方に合わせる
        steps = min(self.lookahead_steps, len(self.latest_preds[0]))
        
        for k in range(steps):
            # AIがこの行動を続けた場合の k 歩先の未来位置
            agent_future = current_pos + action * (k + 1)
            agent_future = np.clip(agent_future, -2.0, 2.0)
            
            # 各ジャマーの k 歩先の未来位置と比較
            for jam_pred in self.latest_preds:
                jam_future = np.array(jam_pred[k])
                dist = np.linalg.norm(agent_future - jam_future)
                
                # 1歩でもセーフティマージンを割り込む未来があれば「危険(False)」
                if dist < self.safety_margin:
                    return False
                    
        return True # すべての未来で距離が保たれていれば「安全(True)」

    def _generate_candidates(self, action):
        """元の行動ベクトルを左右に少しずつ回転させた回避候補リストを作る"""
        candidates = []
        action_norm = np.linalg.norm(action)
        if action_norm < 1e-5:
            return candidates # そもそも止まっているなら候補は出さない
            
        # 左右に30度、60度、90度、120度ずらしたベクトルを生成
        angles = [math.radians(deg) for deg in [30, -30, 60, -60, 90, -90, 120, -120]]
        
        for angle in angles:
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            # 2次元ベクトルの回転行列
            rot_x = action[0]*cos_a - action[1]*sin_a
            rot_y = action[0]*sin_a + action[1]*cos_a
            candidates.append(np.array([rot_x, rot_y], dtype=np.float32))
            
        return candidates