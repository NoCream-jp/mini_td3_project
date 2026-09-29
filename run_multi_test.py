import datetime
import os
import csv
import numpy as np

# モデル読み込みのためにTD3をインポート
from stable_baselines3 import TD3

# 自作ファイルインポート
import config
from my_jammer_env import MyJammerEnv
from my_wrappers import (
    TrajectoryPredictionWrapper, 
    SafetyShieldWrapper, 
    VelocityObservationWrapper,
    KalmanPredictionWrapper,
    PotentialFieldShieldWrapper,
    MonteCarloPredictionWrapper
)

from main import learn_td3 

def run_single_test(test_id, model, env):
    """1回分のテストを実行し、CSVに記録して結果(ステータス, ステップ数)を返す"""
    num_jammers = env.unwrapped.num_jammers
    
    # 変更箇所：日時(test_id)を先頭にする
    csv_filename = f"{test_id}_{config.EXP_NAME}_test_log.csv"
    csv_path = os.path.join(config.OUTPUT_DIR, csv_filename)

    final_status = "Timeout"
    steps_taken = config.MAX_STEPS_PER_EPISODE

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        header = ["step", "agent_x", "agent_y"]
        for i in range(num_jammers):
            header.extend([f"j{i}_x", f"j{i}_y"])
        header.append("reward")
        writer.writerow(header)

        obs, info = env.reset(options={"start_pos": config.AGENT_START_POS})

        for i in range(config.MAX_STEPS_PER_EPISODE):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, finish_flag, over_step_flag, info = env.step(action)

            row_data = [i, obs[0], obs[1]]
            for j in range(num_jammers):
                row_data.extend([obs[2 + j*2], obs[3 + j*2]])
            row_data.append(reward)
            writer.writerow(row_data)

            if finish_flag or over_step_flag:
                steps_taken = i + 1
                if reward >= config.GOAL_REWARD:
                    final_status = "Goal"
                elif reward <= config.OBSTACLE_REWARD:
                    final_status = "Collision"
                break

    return final_status, steps_taken


def run_multiple_tests(model, env, num_tests=5):
    """指定された回数テストを回し、統計を出力する"""
    print(f"\n--- テストを {num_tests} 回実行します ---")
    
    results = []
    goal_count = 0
    collision_count = 0
    timeout_count = 0
    total_goal_steps = 0

    # 変更箇所：秒(%S)を削り、分(%M)までのフォーマットにする
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    for i in range(num_tests):
        # 複数回出力されるため、末尾に _01, _02 と連番を付ける
        test_id = f"{now_time}_{i+1:02d}"
        status, steps = run_single_test(test_id, model, env)
        results.append((test_id, status, steps))
        
        print(f"Test {i+1:02d}: {status} (Steps: {steps})")

        if status == "Goal":
            goal_count += 1
            total_goal_steps += steps
        elif status == "Collision":
            collision_count += 1
        else:
            timeout_count += 1

    print("\n=========================================")
    print(f" テスト統計情報 ({num_tests} 回)")
    print("=========================================")
    print(f"・ゴール到達率: {goal_count / num_tests * 100:.1f} % ({goal_count}/{num_tests})")
    print(f"・衝突率      : {collision_count / num_tests * 100:.1f} % ({collision_count}/{num_tests})")
    print(f"・タイムアウト: {timeout_count / num_tests * 100:.1f} % ({timeout_count}/{num_tests})")
    
    if goal_count > 0:
        print(f"・ゴール到達時の平均ステップ数: {total_goal_steps / goal_count:.1f} steps")
    else:
        print("・ゴール到達時の平均ステップ数: N/A (到達なし)")
    print("=========================================\n")


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. 環境の構築 (テストしたい環境構成を記述)
    raw_env = MyJammerEnv()
    env = VelocityObservationWrapper(raw_env)
    env = KalmanPredictionWrapper(env, horizon_steps=8)
    env = PotentialFieldShieldWrapper(env, lookahead_steps=8, safety_margin=0.35, k_rep=0.01)

    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_id = f"{config.EXP_NAME}_{now_time}"

    print(f"\n=========================================")
    print(f" 複数回テスト実行: {config.EXP_NAME} (ID: {run_id})")
    print(f"=========================================\n")

    # 既存のモデルを読み込む場合は、以下にファイル名を指定
    # 新しく学習する場合は LOAD_MODEL_NAME = None と指定
    LOAD_MODEL_NAME = "20260928_1932_simple_td3_model" 
    
    if LOAD_MODEL_NAME:
        model_path = os.path.join(config.OUTPUT_DIR, LOAD_MODEL_NAME)
        print(f"保存済みのモデルを読み込みます: {model_path}")
        model = TD3.load(model_path, env=env)
    else:
        print("新規にモデルの学習を開始します...")
        model, _ = learn_td3(env, now_time)

    # 3. 指定回数のテストを実行
    num_test_runs = 5  # 回数ここで指定
    run_multiple_tests(model, env, num_tests=num_test_runs)

if __name__ == "__main__":
    main()