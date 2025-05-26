"""
plot_cdf.py
-----------
usage:
    python plot_cdf.py         # 画全部混合
    python plot_cdf.py 7       # 只画第 7 个 episode
"""
import sys, numpy as np, matplotlib.pyplot as plt, os

data = np.load("./logs/jct/all_episode_jct.npz", allow_pickle=True)
episode_jcts = data["episode_jcts"]        # object array

if len(sys.argv) == 2:                     # 画单次
    idx = int(sys.argv[1]) - 1             # 第 N 次 → idx=N-1
    jct = episode_jcts[idx]
    title = f"Episode {idx+1} JCT CDF"
else:                                      # 画混合
    jct = np.concatenate(episode_jcts)
    title = "All-Episodes Mixed JCT CDF"

jct = np.sort(jct)
cdf = np.arange(1, len(jct)+1) / len(jct)

plt.figure(figsize=(6,4))
plt.plot(jct, cdf, lw=2)
plt.xlabel("Job Completion Time (h)")
plt.ylabel("CDF")
plt.title(title)
plt.grid(alpha=0.3)
plt.tight_layout()
os.makedirs("./logs/fig", exist_ok=True)
fname = "cdf_mix.png" if len(sys.argv)==1 else f"cdf_ep{idx+1}.png"
plt.savefig(f"./logs/fig/{fname}", dpi=150)
plt.show()
