import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import config

def draw_averaged_step_rewards(exp_name):
    """
    指定した実験名(exp_name)の直近5件のテストログCSVを読み込み、
    ステップごとの報酬の平均と標準偏差を描画する。
    """
    # ターゲットとなるファイルの検索パターン (例: outputs/test_M7_MC_APF_*_log.csv)
    pattern = os.path.join(config.OUTPUT_DIR, f"test_{exp_name}_*_log.csv")
    
    # 名前でソートし、最新の5件を取得
    file_list = sorted(glob.glob(pattern))[-5:]

    if not file_list:
        print(f"エラー: [{exp_name}] のテストログCSVが見つかりません。")
        print(f"検索パス: {pattern}")
        return

    print(f"=== [{exp_name}] の報酬データを集計します ===")
    for f in file_list:
        print(f" - 読み込み: {f}")

    all_runs_rewards = []
    max_steps = 0

    # 1. 各ファイルから最右列の reward を読み込む
    for file in file_list:
        run_rewards = []
        with open(file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            # rewardが右端（最後のインデックス）にあることを確認
            reward_idx = len(header) - 1 
            
            for row in reader:
                if len(row) > reward_idx:
                    try:
                        run_rewards.append(float(row[reward_idx]))
                    except ValueError:
                        continue # 空行や不正なデータはスキップ
        
        all_runs_rewards.append(run_rewards)
        max_steps = max(max_steps, len(run_rewards))

    # 2. パディング処理（途中で衝突して終了したエピソードの長さを揃える）
    padded_rewards = []
    for run_rewards in all_runs_rewards:
        if len(run_rewards) < max_steps:
            # 途中で終わっている場合は最後の報酬値（ペナルティ）を末尾まで引き伸ばす
            last_val = run_rewards[-1] if run_rewards else 0.0
            extended = run_rewards + [last_val] * (max_steps - len(run_rewards))
            padded_rewards.append(extended)
        else:
            padded_rewards.append(run_rewards)

    rewards_array = np.array(padded_rewards)
    
    # 3. ステップごとの平均と標準偏差を計算
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    steps_range = range(max_steps)

    # 4. 描画設定
    plt.figure(figsize=(9, 5))
    
    # 標準偏差の幅を薄い赤で塗りつぶし
    plt.fill_between(steps_range, 
                     mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, 
                     color='red', alpha=0.15, label='Reward Deviation (±1 std)')
    
    # 平均報酬を赤い実線で描画
    plt.plot(steps_range, mean_rewards, color='red', linewidth=2.0, label=f'Mean Step Reward (n={len(file_list)})')

    plt.yscale('symlog', linthresh=10.0)
    
    plt.ylim(-1200, 100)
    
    plt.title(f"Step-by-Step Reward Analysis ({exp_name})")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Instantaneous Reward (Symlog Scale)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axhline(0, color='black', linewidth=1.0, linestyle='--', zorder=1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    output_filename = f"averaged_test_reward_{exp_name}.png"
    img_path = os.path.join(config.OUTPUT_DIR, output_filename)
    plt.savefig(img_path)
    plt.close()
    print(f"★ステップ報酬の平均化グラフを保存しました: {img_path}\n")


if __name__ == "__main__":
    # グラフ化したい実験名を指定して実行する
    # 例: "M6_Kalman", "M7_MC_APF" など
    draw_averaged_step_rewards(config.TARGET_NAME)