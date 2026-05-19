import os

OUTPUT_DIR = "outputs"

TOTAL_EPISODES = 600            # 学習する総エピソード数
MAX_STEPS_PER_EPISODE = 600     # 1エピソードの最大ステップ数

AGENT_START_POS = [1.9, 1.9]    # テスト時のエージェントのスタート位置

# Jammerを複数定義するためのリスト(初期位置とスピードをそれぞれ設定)
## angleはpsiとしてJammerStateで扱われる値(右0°で　0 <= angle <= 360)。こちらでは度数法で指定する。呼び出すreset側でラジアンに変換している。
JAMMER_CONFIGS = [
    {"pos": [1.0, 1.0], "speed": 0.02, "angle": 0},
    {"pos": [0.0, 0.5], "speed": 0.02, "angle": 0},
    {"pos": [1.0, 0.5], "speed": 0.02, "angle": 0},
    {"pos": [1.0, 0.0], "speed": 0.02, "angle": 0}
]

OBSTACLE_RADIUS = 0.2           # 障害物の判定半径
GOAL_TOLERANCE = 0.1            # ゴール判定の距離

# ----- 報酬　-----
WALL_PENALTY = -100  # 壁に衝突した際の報酬
OBSTACLE_REWARD  = -1000 # 障害物に激突
GOAL_REWARD = 1000 # ゴールに到達したとき
