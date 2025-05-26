import pandas as pd
import os

# 指定CSV路径
csv_path = os.path.expanduser("~/starburst/RLscheduler/Pareto_curves/budget_factors/all_runs.csv")

# 读取CSV
df = pd.read_csv(csv_path)

# 只保留 method 为 "RL" 的行
filtered_df = df[df["method"] == "RL"]

# 写回原文件（覆盖）
filtered_df.to_csv(csv_path, index=False)
