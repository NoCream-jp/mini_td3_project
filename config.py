import os

OUTPUT_DIR = "outputs"

TOTAL_EPISODES = 300            # 学習する総エピソード数
MAX_STEPS_PER_EPISODE = 200     # 1エピソードの最大ステップ数

AGENT_START_POS = [1.9, 1.9]    # テスト時のエージェントのスタート位置

# Jammerを複数定義するためのリスト(初期位置とスピードをそれぞれ設定)
JAMMER_CONFIGS = [
    {"pos": [1.0, 1.0], "speed": 0.2},     # 1体目: 半径約1.4を時計回り
    {"pos": [-0.5, -0.5], "speed": -0.15}  # 2体目: 半径約0.7を反時計回り
]

OBSTACLE_RADIUS = 0.4           # 障害物の判定半径
GOAL_TOLERANCE = 0.1            # ゴール判定の距離

# ----- 報酬　-----
WALL_PENALTY = -100  # 壁に衝突した際の報酬
OBSTACLE_REWARD  = -1000
GOAL_REWARD = 1000
