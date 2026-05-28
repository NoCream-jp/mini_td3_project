import os

OUTPUT_DIR = "outputs"

TOTAL_EPISODES = 500            # 学習する総エピソード数
MAX_STEPS_PER_EPISODE = 400     # 1エピソードの最大ステップ数

AGENT_START_POS = [2.0, 2.0]    # テスト時のエージェントのスタート位置
GOAL_POS = [-2.0, -2.0]         # エージェントが目指す終点

# Jammerの設定
# type: "sin_wave" を新設。
# 指定したスタート[2.0, 0]から左[-2.0, 0]へ向かってパトロールしつつ、Y軸方向に波形に揺れる
JAMMER_CONFIGS = [
    {
        "type": "sin_wave", 
        "start_pos": [2.0, 0.0],   # 波の基準線の開始点
        "end_pos": [-2.0, 0.0],    # 波の基準線の終着点
        "amplitude": 0.8,          # 波の振幅（Y軸方向にどれくらい大きく揺れるか）
        "frequency": 4.0,          # 波の周波数（往復の間に何サイクル波を作るか）
        "speed": 0.1,             # 移動スピード
    }
]

OBSTACLE_RADIUS = 0.2           # 障害物の判定半径
GOAL_TOLERANCE = 0.1            # ゴール判定の距離

# ----- 報酬 -----
WALL_PENALTY = -100  # 壁に衝突した際の報酬
OBSTACLE_REWARD  = -1000 # 障害物に激突
GOAL_REWARD = 1000 # ゴールに到達したとき