# -*- coding: utf-8 -*-
"""自主实践③：scatter 二维投影（任选两特征，按簇着色）

用两个最能拉开差距的特征组合各画一张：
  (a) 月均消费 × 持有产品数
  (b) 持有产品数 × 逾期次数
簇中心先逆标准化回原始坐标，再画星形标记。

运行：python generated/07_scatter_2d.py
产出：outputs/07_scatter_2d.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "bank_customers.csv"
OUT = BASE / "outputs"

FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
NAMES = {0: "高消费低黏性", 1: "低频沉睡", 2: "高余额高逾期", 3: "多产品稳健"}
COLORS = ["#2f6fed", "#868e96", "#e8590c", "#2f9e44"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    df = pd.read_csv(DATA)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATS])
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    df["cluster"] = km.labels_

    # 簇中心在标准化空间里，逆变换才画得到原始坐标轴上
    centers = scaler.inverse_transform(km.cluster_centers_)
    centers = pd.DataFrame(centers, columns=FEATS)

    pairs = [("avg_monthly_spend", "product_count"), ("product_count", "overdue_count")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (fx, fy) in zip(axes, pairs):
        for cid in range(4):
            sub = df[df["cluster"] == cid]
            ax.scatter(sub[fx], sub[fy], s=26, alpha=0.8,
                       color=COLORS[cid], label=f"簇{cid} {NAMES[cid]}（{len(sub)}人）")
        ax.scatter(centers[fx], centers[fy], marker="*", s=380,
                   edgecolor="black", facecolor="gold", zorder=5, label="簇中心")
        ax.set_xlabel({"avg_monthly_spend": "月均消费（元）",
                       "product_count": "持有产品数（个）",
                       "overdue_count": "逾期次数（次）"}[fx])
        ax.set_ylabel({"avg_monthly_spend": "月均消费（元）",
                       "product_count": "持有产品数（个）",
                       "overdue_count": "逾期次数（次）"}[fy])
        ax.grid(alpha=0.3)
    axes[0].set_title("（a）月均消费 × 持有产品数")
    axes[1].set_title("（b）持有产品数 × 逾期次数")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("自主实践③：二维投影下的四簇边界（标准化 KMeans，k=4，random_state=42）")
    fig.tight_layout()
    out = OUT / "07_scatter_2d.png"
    fig.savefig(out, dpi=150)
    print(f"已写出：{out}")


if __name__ == "__main__":
    main()
