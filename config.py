import os

OUTPUT_DIR = "outputs"

TOTAL_EPISODES = 500            # 学習する総エピソード数
MAX_STEPS_PER_EPISODE = 300     # 1エピソードの最大ステップ数

AGENT_START_POS = [2.0, 2.0]    # テスト時のエージェントのスタート位置
GOAL_POS = [-2.0, -2.0]         # エージェントが目指す終点

# Jammerの設定
# type: "linear_cross" を新設。右下(2,-2)から左上(-2,2)へ直進し、往復パトロールする
JAMMER_CONFIGS = [
    {
        "type": "linear_cross", 
        "pos": [2.0, -2.0],     # 初期位置（右下）
        "speed": 0.1,           # エージェントの最高速度(0.1)と完全に同期
    }
]

OBSTACLE_RADIUS = 0.2           # 障害物の判定半径
GOAL_TOLERANCE = 0.1            # ゴール判定の距離

# ----- 報酬　-----
WALL_PENALTY = -100  # 壁に衝突した際の報酬
OBSTACLE_REWARD  = -1000 # 障害物に激突
GOAL_REWARD = 1000 # ゴールに到達したとき