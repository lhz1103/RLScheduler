#!/usr/bin/env python3
# ── plot_curves_cost_gpu_hours.py ────────────────────────────────
"""
读取  ~/starburst/RLscheduler/Pareto_curves/all_runs.csv
为每个 method 画一条 Cost–JCT 折线
横轴: total_cloud_cost  (GPU·hours)
纵轴: mean_jct         (hours)
输出: pareto_curves_gpuh.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1) 读取实验结果
csv_path = os.path.expanduser(
    "~/starburst/RLscheduler/Pareto_curves/all_runs.csv")
df = pd.read_csv(csv_path)

# 如果你只想用稳态 episode，可在这里过滤：
# df = df[df["episode"] >= 400]

# ------------------------------------------------------------------
# 2) 每 (method,param) 取平均，降低点数
agg = (df.groupby(["method", "param"])
         .agg(cost=("cost", "mean"),        # cost 本身就是 GPU·h
              jct =("mean_jct", "mean"))
         .reset_index())

# ------------------------------------------------------------------
# 3) 绘制多条折线
plt.figure(figsize=(7,5))
colors  = ["tab:orange", "tab:green", "tab:purple",
           "tab:red", "tab:blue", "tab:brown", "tab:pink"]
markers = ["s", "D", "o", "^", "X", "v", "P"]

for i, (m, g) in enumerate(agg.groupby("method")):
    g_sorted = g.sort_values("cost")          # 按 GPU·h 升序
    plt.plot(g_sorted["cost"], g_sorted["jct"],
             marker=markers[i % len(markers)],
             color=colors[i % len(colors)],
             linewidth=1.5, markersize=6,
             label=m.replace("_", " "))

# ------------------------------------------------------------------
# 4) 细节
plt.xlabel("Cloud Cost  (GPU·hours)")
plt.ylabel("Avg. JCT  (hours)")
plt.title("Cost–JCT Trade‑off Curves")
plt.grid(alpha=0.3, ls="--", zorder=0)
plt.legend()
plt.tight_layout()

# ------------------------------------------------------------------
# 5) 保存 & 展示
out_png = os.path.join(os.path.dirname(csv_path),
                       "pareto_curves_gpuh.png")
plt.savefig(out_png, dpi=150)
print("figure saved to:", out_png)
