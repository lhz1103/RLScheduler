#!/usr/bin/env python3
# ── plot_pareto.py ────────────────────────────────────────────────
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

# 1. 读取 CSV ------------------------------------------------------
csv_path = os.path.expanduser(
    "~/starburst/RLscheduler/Pareto_curves/all_runs.csv"
)
df = pd.read_csv(csv_path)

# 如你只想取最后 100 episode 的稳态结果，可在此过滤
# df = df[df["episode"] >= 400]

# 2. 先对同一 (method,param) 求平均，降低点数 ----------------------
agg = (df.groupby(["method", "param"])
         .agg(cost=("cost", "mean"),
              jct =("mean_jct", "mean"))
         .reset_index())

# 3. 帕累托前沿：cost、jct 都越小越好 ----------------------------
def pareto_frontier(data, x="cost", y="jct"):
    data = data.sort_values(x, ascending=True).reset_index()
    best_y = np.inf
    idx_front = []
    for i, row in data.iterrows():
        if row[y] < best_y - 1e-9:     # 严格更优
            best_y = row[y]
            idx_front.append(row["index"])
    return idx_front

front_idx = pareto_frontier(agg, "cost", "jct")

# 4. 画散点 + 帕累托折线 -------------------------------------------
plt.figure(figsize=(7,5))

markers = ["o", "s", "^", "v", "D", "P", "X"]
colour_map = {}

for i, (m, group) in enumerate(agg.groupby("method")):
    colour_map[m] = f"C{i}"
    plt.scatter(group["cost"], group["jct"],
                marker=markers[i%len(markers)],
                color=f"C{i}", alpha=0.6,
                label=m)

# 前沿：按 cost 排序后连线
frontier = agg.loc[front_idx].sort_values("cost")
plt.plot(frontier["cost"], frontier["jct"],
         "-x", c="red", lw=2, label="Pareto front")

# 5. 修饰 ----------------------------------------------------------
plt.xlabel("Cloud cost (GPU·hours)")
plt.ylabel("Mean Job Completion Time (hours)")
plt.title("Cost vs. JCT Pareto Frontier")
plt.grid(alpha=0.3, ls="--")
plt.legend()
plt.tight_layout()

# 6. 保存 & 显示 ---------------------------------------------------
out_png = os.path.join(os.path.dirname(csv_path), "pareto_frontier.png")
plt.savefig(out_png, dpi=150)
print("figure saved to:", out_png)
