import math
import gymnasium as gym
import numpy as np
from collections import deque
from typing import cast

import config
from my_jammer_env import MyJammerEnv

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# =====================================================================
# 1. 速度ベクトルを渡す学習ラッパー
# =====================================================================
class VelocityObservationWrapper(gym.Wrapper):
    """
    環境から出力される観測(obs)に、「各ジャマーとの相対速度ベクトル」を追加して
    AIのニューラルネットワークに渡すための視覚拡張ラッパー。
    """
    def __init__(self, env):
        super().__init__(env)
        
        raw_env = self.env.unwrapped
        self.num_jammers = len(config.JAMMER_CONFIGS)
        
        old_dim = 2 + (self.num_jammers * 2)
        new_dim = old_dim + (self.num_jammers * 2)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32
        )
        
        # None を使わず、最初から安全なゼロ配列（ダミー）を入れておく
        self.prev_ego_pos = np.zeros(2, dtype=np.float32)
        self.prev_jam_positions = [np.zeros(2, dtype=np.float32) for _ in range(self.num_jammers)]

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # --- 初期位置の記録 ---
        self.prev_ego_pos = obs[0:2]  # 0,1番目はエージェント
        self.prev_jam_positions = []
        
        for i in range(self.num_jammers):
            # ジャマーのx, yは2番目以降に2つずつ格納されている
            jx_jy = obs[2 + i*2 : 4 + i*2]
            self.prev_jam_positions.append(jx_jy)
            
        # リセット直後はまだ動いていないため、相対速度はすべて 0.0
        rel_vs = np.zeros(self.num_jammers * 2, dtype=np.float32)
        
        # 元の観測配列の後ろに、相対速度の配列を連結（くっつける）
        new_obs = np.concatenate([obs, rel_vs], dtype=np.float32)
        
        return new_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- 1. エージェントの速度計算 ---
        current_ego_pos = obs[0:2]
        ego_v = current_ego_pos - self.prev_ego_pos
        
        rel_vs = []
        current_jam_positions = []
        
        # --- 2. 各ジャマーの相対速度計算 ---
        for i in range(self.num_jammers):
            # 現在のジャマー位置を取得
            jam_pos = obs[2 + i*2 : 4 + i*2]
            
            # ジャマーの絶対速度
            jam_v = jam_pos - self.prev_jam_positions[i]
            
            # 相対速度 = 相手の速度 - 自分の速度
            rel_v = jam_v - ego_v
            rel_vs.extend(rel_v)
            
            # 次のステップのために現在の位置を保存リストへ
            current_jam_positions.append(jam_pos)
            
        # --- 3. 過去の記録を更新 ---
        self.prev_ego_pos = current_ego_pos
        self.prev_jam_positions = current_jam_positions
        
        # --- 4. 新しい観測配列の作成 ---
        rel_vs_array = np.array(rel_vs, dtype=np.float32)
        new_obs = np.concatenate([obs, rel_vs_array], dtype=np.float32)
        
        return new_obs, reward, terminated, truncated, info


# =====================================================================
# 2. 単純な軌道予測ラッパー (The Oracle)
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
                # 特定の軌道に依存せず、開始直後（履歴不足時）は「停止」として扱う
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
# 3. カルマンフィルタ定義、それを利用する予測ラッパー
# =====================================================================
class KalmanFilter2D:
    """2次元の等速直線運動（CVモデル）用カルマンフィルタ"""
    def __init__(self, dt=1.0, std_acc=0.03, std_meas=0.001):
        self.dt = dt
        # 状態ベクトル: [x, y, vx, vy]^T
        self.x = np.zeros(4, dtype=np.float32)
        
        # 状態遷移行列 F (未来を予測する物理法則: 等速直線)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 観測行列 H (センサーから見えるもの＝位置 x, y のみ)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 誤差共分散行列 P (現在の推定に対する「自信のなさ」)
        self.P = np.eye(4, dtype=np.float32) * 1.0
        
        # 観測ノイズ共分散 R (センサーのブレ具合)
        self.R = np.eye(2, dtype=np.float32) * (std_meas**2)
        
        # プロセスノイズ共分散 Q (想定外の動き・加速度のブレ具合)
        # サイン波のような「直線から外れる動き」を許容するための重要なパラメータ
        self.Q = np.eye(4, dtype=np.float32) * (std_acc**2)
        
    def predict(self):
        """事前推定：現在の速度のまま進んだらどこにいるか"""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
    def update(self, z):
        """事後推定：実際の観測データ(z)を使って予測を修正する"""
        y = z - np.dot(self.H, self.x) # 予測と実際のズレ（イノベーション）
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # カルマンゲイン
        
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

import gymnasium as gym

class KalmanPredictionWrapper(gym.Wrapper):
    """
    カルマンフィルタを用いてジャマーの未来軌道を予測するラッパー。
    以前の TrajectoryPredictionWrapper の完全上位互換です。
    """
    def __init__(self, env, horizon_steps=20):
        super().__init__(env)
        self.horizon_steps = horizon_steps
        
        raw_env = self.env.unwrapped
        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        self.num_jammers = raw_env.num_jammers
        self.kfs = []

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        self.kfs = []
        for i in range(self.num_jammers):
            # カルマンフィルタを初期化
            kf = KalmanFilter2D(dt=1.0, std_acc=0.03, std_meas=0.001)
            
            jx = obs[2 + i*2]
            jy = obs[3 + i*2]
            
            # 初期位置をセット、初期速度は0のままスタート
            kf.x[0] = jx
            kf.x[1] = jy
            self.kfs.append(kf)
            
        info['jam_preds'] = self._predict_trajectories()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 各ジャマーのカルマンフィルタを更新
        for i in range(self.num_jammers):
            jx = obs[2 + i*2]
            jy = obs[3 + i*2]
            z = np.array([jx, jy], dtype=np.float32)
            
            # 1. 前回の状態から今の位置を「予測」
            self.kfs[i].predict()
            
            # 2. 実際の今の位置(z)を使って、内部の速度ベクトルを「修正」
            self.kfs[i].update(z)
            
        # 修正された最新の速度ベクトルを使って未来軌道を生成
        info['jam_preds'] = self._predict_trajectories()
        return obs, reward, terminated, truncated, info

    def _predict_trajectories(self):
        all_preds = []
        for kf in self.kfs:
            pred_traj = []
            
            # 未来をシミュレーションするために、現在の状態(x, y, vx, vy)をコピー
            sim_x = np.copy(kf.x)
            
            for _ in range(self.horizon_steps):
                # 状態遷移行列 F を掛けて、1歩未来へ進める
                sim_x = np.dot(kf.F, sim_x)
                
                # 世界の果て（壁）での反射ロジック
                if sim_x[0] < -2.0 or sim_x[0] > 2.0:
                    sim_x[2] = -sim_x[2] # X方向の速度ベクトルを反転
                    sim_x[0] = np.clip(sim_x[0], -2.0, 2.0)
                if sim_x[1] < -2.0 or sim_x[1] > 2.0:
                    sim_x[3] = -sim_x[3] # Y方向の速度ベクトルを反転
                    sim_x[1] = np.clip(sim_x[1], -2.0, 2.0)
                    
                pred_traj.append((sim_x[0], sim_x[1]))
                
            all_preds.append(pred_traj)
            
        return all_preds

# =====================================================================
# 4. モンテカルロ法を利用した予測ラッパー 
# =====================================================================
class MonteCarloPredictionWrapper(gym.Wrapper):
    """
    モンテカルロ法でジャマーの未来軌道を予測するラッパー。
    各ジャマーについて複数サンプルの軌道を生成し、その平均を予測とする。
    """

    def __init__(self, env, horizon_steps=20, num_samples=30, noise_std=0.05):
        super().__init__(env)
        self.horizon_steps = horizon_steps
        self.num_samples = num_samples
        self.noise_std = noise_std

        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        self.num_jammers = raw_env.num_jammers

        # 前ステップ位置（速度推定用）
        self.prev_positions = [np.zeros(2, dtype=np.float32) for _ in range(self.num_jammers)]

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.prev_positions = []
        for i in range(self.num_jammers):
            pos = obs[2 + i*2 : 4 + i*2]
            self.prev_positions.append(pos)

        info['jam_preds'] = self._predict(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info['jam_preds'] = self._predict(obs)

        # 更新
        for i in range(self.num_jammers):
            self.prev_positions[i] = obs[2 + i*2 : 4 + i*2]

        return obs, reward, terminated, truncated, info

    def _predict(self, obs):
        all_preds = []

        for i in range(self.num_jammers):
            current_pos = obs[2 + i*2 : 4 + i*2]
            prev_pos = self.prev_positions[i]

            # 速度推定（単純差分）
            velocity = current_pos - prev_pos

            samples = []

            for _ in range(self.num_samples):
                traj = []
                sim_pos = np.copy(current_pos)
                sim_vel = np.copy(velocity)

                for _ in range(self.horizon_steps):
                    # ランダムノイズ追加（ここがモンテカルロ）
                    noise = np.random.normal(0, self.noise_std, size=2)
                    sim_vel = sim_vel + noise

                    sim_pos = sim_pos + sim_vel

                    # 壁反射
                    for d in range(2):
                        if sim_pos[d] < -2.0 or sim_pos[d] > 2.0:
                            sim_vel[d] = -sim_vel[d]
                            sim_pos[d] = np.clip(sim_pos[d], -2.0, 2.0)

                    traj.append(sim_pos.copy())

                samples.append(traj)

            # サンプル平均を取る
            mean_traj = []
            for t in range(self.horizon_steps):
                mean_pos = np.mean([samples[s][t] for s in range(self.num_samples)], axis=0)
                mean_traj.append((mean_pos[0], mean_pos[1]))

            all_preds.append(mean_traj)

        return all_preds

# =====================================================================
# 5. 30度ずつ回転を試す、回避ラッパー (The Shield)
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
            # reward = float(reward) - 5.0 # 「助けてもらうような危険な行動をするな」という教育的指導を消去。比較のため。
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

# =====================================================================
# 6. 人工ポテンシャル法で滑らかに離れる回避ラッパー
# =====================================================================
class PotentialFieldShieldWrapper(gym.Wrapper):
    """
    人工ポテンシャル法（APF）を用いた安全シールド。
    AIの行動を「引力」、予測されるジャマーからの危険度を「斥力」とし、
    それらの合力ベクトルを計算して滑らかに回避する。
    latest_preds：jammerの未来のデータ
    """
    def __init__(self, env, lookahead_steps=15, safety_margin=0.35, k_rep=0.05):
        super().__init__(env)
        self.lookahead_steps = lookahead_steps
        self.safety_margin = safety_margin
        
        # 斥力の強さを決めるパラメータ（大きいほど強く弾かれる）
        self.k_rep = k_rep 
        
        self.latest_preds = []

    def reset(self, seed=None, options=None):
        # envのresetの戻り値の配列のほうを、latest_predsに格納しておく
        obs, info = self.env.reset(seed=seed, options=options)
        self.latest_preds = info.get('jam_preds', [])
        return obs, info

    def step(self, action):
        # 1. ポテンシャル法による合力（安全な行動ベクトル）の計算
        safe_action = self._calculate_apf_action(action)
        
        # 2. 環境を進める
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        self.latest_preds = info.get('jam_preds', [])
        
        # 3. シールド介入判定
        if not np.allclose(action, safe_action, atol=1e-5):
            info['shield_activated'] = True
        else:
            info['shield_activated'] = False
            
        return obs, reward, terminated, truncated, info

    def _calculate_apf_action(self, action):
        if not self.latest_preds:
            return action
            
        # MyJammerEnv型にキャストしないとエラーが出る
        raw_env = cast(MyJammerEnv, self.env.unwrapped)
        current_pos = raw_env.location
        
        # 引力ベクトル：AIが本来行きたい方向（TD3の出力）
        F_att = action 
        # 斥力ベクトルの初期化
        F_rep = np.zeros(2, dtype=np.float32)
        
        is_danger = False
        action_norm = np.linalg.norm(action)
        
        if action_norm < 1e-6:
            return action # 止まっているなら何もしない
            
        steps = min(self.lookahead_steps, len(self.latest_preds[0]))
        
        # 影響圏（この距離以内に入ったら斥力が発生し始める）
        influence_radius = self.safety_margin + 0.2
        
        for k in range(steps):
            # AIがそのまま進んだ場合の未来位置
            agent_future = current_pos + action * (k + 1)
            
            for jam_pred in self.latest_preds:
                jam_future = np.array(jam_pred[k])
                dist = np.linalg.norm(agent_future - jam_future)
                
                # もし影響圏内に入る未来があれば、斥力を計算して足し合わせる
                if dist < influence_radius:
                    is_danger = True
                    # ジャマーからエージェントへ向かう方向ベクトル
                    diff = agent_future - jam_future
                    dir_vector = diff / (dist + 1e-6)
                    
                    # 距離が近いほど指数関数的に強くなる斥力の大きさ
                    magnitude = self.k_rep * ((1.0 / (dist + 1e-6)) - (1.0 / influence_radius))
                    F_rep += dir_vector * magnitude

        # 危険が全く予測されなかった場合は、本来のAIの行動をそのまま通す
        if not is_danger:
            return action

        # 【ポテンシャル場の合力計算】
        F_total = F_att + F_rep
        total_norm = np.linalg.norm(F_total)

        # AIが出力した本来の行動の「速さ（ベクトルの長さ）」は維持しつつ、
        # 合力によって「向き」だけを滑らかに曲げる処理
        if 1e-5 < total_norm :
            safe_action = (F_total / total_norm) * action_norm
            return safe_action
        else:
            return np.zeros(2, dtype=np.float32) # 合力がゼロ（完全に相殺）なら急停止
