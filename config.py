import os

OUTPUT_DIR = "outputs"

# 学習する総エピソード数
TOTAL_EPISODES = 500
# 1エピソードの最大ステップ数
MAX_STEPS_PER_EPISODE = 500

# テスト時のエージェントのスタート位置
AGENT_START_POS = [2.0, 2.0]
# エージェントが目指す終点
GOAL_POS = [-2.0, -2.0]

# Jammerの設定
# type: "sin_wave" を新設。
# 指定したスタート[2.0, 0]から左[-2.0, 0]へ向かってパトロールしつつ、Y軸方向に波形に揺れる
"""
SAMPLE_JAMMER_CONFIGS = [
    {
        # ① 直線運動（start_pos と end_pos が必須）
        "type": "linear_cross", 
        "start_pos": [-2.0, -1.0], # 左下から
        "end_pos": [2.0, -1.0],    # 右下へ横断
        "speed": 0.04,
    },
    {
        # ② サイン波（直線設定 ＋ amplitude と frequency が必須）
        "type": "sin_wave", 
        "start_pos": [2.0, 0.0],   # 右端から
        "end_pos": [-2.0, 0.0],    # 左端へ
        "amplitude": 0.8,          # 揺れ幅の大きさ
        "frequency": 4.0,          # 波の数
        "speed": 0.06,
    },
    {
        # ③ 円運動（center と size と angle が必須）
        "type": "circle",
        "center": [0.0, 1.0],
        "size": 0.7,
        "speed": 0.05,
        "angle": 0.0
    }
]
"""
JAMMER_CONFIGS = [
    {
        "type": "circle",
        "center": [0.0, 0.0],
        "size": 0.7,
        "speed": 1.0,
        "angle": 0.0
    },
]

OBSTACLE_RADIUS = 0.2           # 障害物の判定半径
GOAL_TOLERANCE = 0.1            # ゴール判定の距離

# ----- 報酬 -----
WALL_PENALTY = -100  # 壁に衝突した際の報酬
OBSTACLE_REWARD  = -1000 # 障害物に激突
GOAL_REWARD = 1000 # ゴールに到達したとき