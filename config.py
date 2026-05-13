import os

# --- 1. ディレクトリ設定 ---
OUTPUT_DIR = "outputs"

# --- 2. 学習ハイパーパラメータ ---
TOTAL_EPISODES = 300            # 学習する総エピソード数
MAX_STEPS_PER_EPISODE = 200     # 1エピソードの最大ステップ数

# --- 3. 環境パラメータ ---
# ジャマーの軌道はenvファイルで定義
AGENT_START_POS = [1.9, 1.9]    # テスト時のエージェントのスタート位置
JAMMER_START_POS = [1.0, 1.0]   # ジャマー（障害物）の初期位置
JAMMER_SPEED = 0.2             # ジャマーの移動速度
OBSTACLE_RADIUS = 0.4           # 障害物の判定半径
GOAL_TOLERANCE = 0.1            # ゴール判定の距離