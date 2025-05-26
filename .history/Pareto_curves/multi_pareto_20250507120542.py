#!/usr/bin/env python3
import os, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from itertools import cycle

csv_path = os.path.expanduser("~/starburst/RLscheduler/Pareto_curves/all_runs.csv")
df = pd.read_csv(csv_path)

# 平均每个 (method,param)
agg = (df.groupby(["method","param"])
         .agg(cost=("cost","mean"), jct=("mean_jct","mean"))
         .reset_index())

# ------------------ 动态颜色 & marker -------------------------------
methods = agg["method"].unique()
n_meth  = len(methods)

# ① 颜色：tab20 可给 20 种互异颜色；>20 时再循环
cmap = mpl.colormaps["tab20"]
color_cycle  = [cmap(i) for i in range(20)]
colors = {m: color_cycle[i % 20] for i, m in enumerate(methods)}

# ② marker：自定义 12 个形状；>12 时循环
marker_set = ["o","s","D","^","v","P","X","*","h","8","<",">"]
markers = {m: marker_set[i % len(marker_set)] for i, m in enumerate(methods)}

# ------------------ 画图 --------------------------------------------
plt.figure(figsize=(7,5))

for m, g in agg.groupby("method"):
    g = g.sort_values("cost")
    plt.plot(g["cost"], g["jct"],
             marker=markers[m], color=colors[m],
             linewidth=1.8, markersize=6, label=m.replace("_"," "))

plt.xlabel("Cloud Cost (GPU·hours)")
plt.ylabel("Avg. JCT (hours)")
plt.title("Cost–JCT Trade‑off Curves")
plt.grid(alpha=0.3, ls="--", zorder=0)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()

out_png = os.path.join(os.path.dirname(csv_path), "pareto_curves_gpuh.png")
plt.savefig(out_png, dpi=150)
print("saved:", out_png)
