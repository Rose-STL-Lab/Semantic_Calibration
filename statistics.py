import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "./metrics_log.csv"   # <- 改成你的 csv 路径

df = pd.read_csv(CSV_PATH)

# 基础清理：按 step 排序；把关键列转成数值（防止读成字符串）
df["step"] = pd.to_numeric(df["step"], errors="coerce")
df = df.dropna(subset=["step"]).sort_values("step")

def to_num(col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 你关心的列（按你的表头）
cols = [
    "rewards/equiv_loss_reward/mean",
    "rewards/correctness_reward/mean",
    "reward_std",
    "frac_reward_zero_std",
    "kl",
    "reward",
]
for c in cols:
    to_num(c)

import os

OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)

def save_show(fig_name: str):
    path = os.path.join(OUTDIR, fig_name)
    plt.savefig(path, dpi=200)
    print("[saved]", path)
    plt.show()

# ====== (可选) 把你前面 Fig1-3 也保存下来：在每个 plt.show() 前加一行 save_show(...) ======
# 如果你不想改上面的代码，就在这里重画一遍并保存（最省心）

# -----------------------
# Fig 1 (save): Equiv vs Correctness (mean)
# -----------------------
plt.figure()
if "rewards/equiv_loss_reward/mean" in df.columns:
    plt.plot(df["step"], df["rewards/equiv_loss_reward/mean"], label="equiv_loss_reward/mean")
if "rewards/correctness_reward/mean" in df.columns:
    plt.plot(df["step"], df["rewards/correctness_reward/mean"], label="correctness_reward/mean")
plt.xlabel("step")
plt.ylabel("mean reward")
plt.title("Equiv vs Correctness Reward (mean)")
plt.legend()
plt.tight_layout()
save_show("fig1_equiv_vs_correctness_mean.png")

# -----------------------
# Fig 2 (save): Reward dispersion indicators
# -----------------------
plt.figure()
if "reward_std" in df.columns:
    plt.plot(df["step"], df["reward_std"], label="reward_std")
if "frac_reward_zero_std" in df.columns:
    plt.plot(df["step"], df["frac_reward_zero_std"], label="frac_reward_zero_std")
plt.xlabel("step")
plt.ylabel("value")
plt.title("Reward Dispersion Indicators")
plt.legend()
plt.tight_layout()
save_show("fig2_reward_dispersion.png")

# -----------------------
# Fig 3 (save): Reward over training (single metric)
# -----------------------
plt.figure()
if "reward" in df.columns:
    plt.plot(df["step"], df["reward"], label="reward")
plt.xlabel("step")
plt.ylabel("reward")
plt.title("Total Reward Over Training")
plt.legend()
plt.tight_layout()
save_show("fig3_reward_over_training.png")

# -----------------------
# Fig 4 (save): KL over training (single metric)
# -----------------------
plt.figure()
if "kl" in df.columns:
    plt.plot(df["step"], df["kl"], label="kl")
plt.xlabel("step")
plt.ylabel("kl")
plt.title("KL Over Training")
plt.legend()
plt.tight_layout()
save_show("fig4_kl_over_training.png")


# =========================================================
# Fig 5: Smoothed means ——
#   add: total reward (smoothed)
# =========================================================
W = 20

if "rewards/correctness_reward/mean" in df.columns:
    df["corr_smooth"] = df["rewards/correctness_reward/mean"].rolling(W, min_periods=1).mean()
if "rewards/equiv_loss_reward/mean" in df.columns:
    df["equiv_smooth"] = df["rewards/equiv_loss_reward/mean"].rolling(W, min_periods=1).mean()
if "reward" in df.columns:
    df["reward_smooth"] = df["reward"].rolling(W, min_periods=1).mean()

plt.figure()
if "reward_smooth" in df.columns:
    plt.plot(df["step"], df["reward_smooth"], label=f"total reward (MA{W})")
if "equiv_smooth" in df.columns:
    plt.plot(df["step"], df["equiv_smooth"], label=f"equiv mean (MA{W})")
if "corr_smooth" in df.columns:
    plt.plot(df["step"], df["corr_smooth"], label=f"correctness mean (MA{W})")

plt.xlabel("step")
plt.ylabel("smoothed value")
plt.title("Smoothed Reward Trends")
plt.legend()
plt.tight_layout()
save_show("fig5_smoothed_reward_trends.png")

import numpy as np

def compute_slope(x, y):
    """
    返回线性回归斜率 a（y = a x + b）
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    a, b = np.polyfit(x, y, 1)
    return a

# === 计算 equiv / correctness / reward 的斜率 ===
if "equiv_smooth" in df.columns:
    equiv_slope = compute_slope(df["step"].values, df["equiv_smooth"].values)
    print(f"[Slope] equiv_smooth slope = {equiv_slope:.3e}")

if "corr_smooth" in df.columns:
    corr_slope = compute_slope(df["step"].values, df["corr_smooth"].values)
    print(f"[Slope] correctness_smooth slope = {corr_slope:.3e}")

if "reward_smooth" in df.columns:
    reward_slope = compute_slope(df["step"].values, df["reward_smooth"].values)
    print(f"[Slope] total_reward_smooth slope = {reward_slope:.3e}")

# =========================================================
# NEW Fig 6: reward_std vs equiv_reward_std —— 谁在提供组内区分度？
# =========================================================
# 你 CSV 里有 rewards/equiv_loss_reward/std，拿来和 reward_std 对比
if "rewards/equiv_loss_reward/std" in df.columns:
    df["rewards/equiv_loss_reward/std"] = pd.to_numeric(df["rewards/equiv_loss_reward/std"], errors="coerce")

plt.figure()
if "reward_std" in df.columns:
    plt.plot(df["step"], df["reward_std"], label="reward_std (total)")
if "rewards/equiv_loss_reward/std" in df.columns:
    plt.plot(df["step"], df["rewards/equiv_loss_reward/std"], label="equiv_loss_reward/std")
plt.xlabel("step")
plt.ylabel("std")
plt.title("Std Comparison: Total Reward vs Equiv Reward")
plt.legend()
plt.tight_layout()
save_show("fig6_std_total_vs_equiv.png")
