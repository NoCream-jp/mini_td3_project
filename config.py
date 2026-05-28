import os

OUTPUT_DIR = "outputs"

TOTAL_EPISODES = 500            # 学習する総エピソード数
MAX_STEPS_PER_EPISODE = 300     # 1エピソードの最大ステップ数

AGENT_START_POS = [2.0, 2.0]    # テスト時のエージェントのスタート位置
GOAL_POS = [-2.0, -2.0]         # エージェントが目指す終点

# Jammerを複数定義するためのリスト
# type: "figure8" (8の字) または "circle" (円)
JAMMER_CONFIGS = [
    {
        "type": "figure8", 
        "center": [0.0, 0.0], # 軌道の中心
        "size": 1.5,          # 軌道の大きさ（振幅）
        "speed": 0.033,        # 動く速さ
        "angle": 0            # 初期位相（スタート位置のズレ）
    }
]

OBSTACLE_RADIUS = 0.2           # 障害物の判定半径
GOAL_TOLERANCE = 0.1            # ゴール判定の距離

# ----- 報酬　-----
WALL_PENALTY = -100  # 壁に衝突した際の報酬
OBSTACLE_REWARD  = -1000 # 障害物に激突
GOAL_REWARD = 1000 # ゴールに到達したとき