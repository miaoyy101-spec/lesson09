# -*- coding: utf-8 -*-
"""独立研究②③：k-均值对环形数据失效 + DBSCAN 对比 + 轮廓系数

② 生成"内圈 + 外圈"玩具数据，k-均值只能按空间位置直线切，必然把两个环切碎；
   DBSCAN 按密度连通，能把两个环各自完整圈出来。
③ 在真实数据上算 silhouette_score，解释它衡量什么。

运行：python generated/08_independent_study.py
产出：outputs/08_ring_dbscan.png、outputs/09_silhouette.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_circles
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "bank_customers.csv"
OUT = BASE / "outputs"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ring_experiment():
    X, y = make_circles(n_samples=400, factor=0.45, noise=0.05, random_state=42)
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
    db = DBSCAN(eps=0.15, min_samples=5).fit(X)
    ari_km = adjusted_rand_score(y, km.labels_)
    ari_db = adjusted_rand_score(y, db.labels_)
    n_noise = int((db.labels_ == -1).sum())
    print(f"环形数据（400 点、真值 2 类）：")
    print(f"  k-均值 k=2 的 ARI = {ari_km:.3f}（切成上下两半，两环都被切碎）")
    print(f"  DBSCAN eps=0.15 min_samples=5 的 ARI = {ari_db:.3f}，噪声点 {n_noise} 个")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=16)
    axes[0].set_title("原始环形数据（真值：内圈/外圈）")
    axes[1].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="coolwarm", s=16)
    axes[1].set_title(f"k-均值 k=2：ARI={ari_km:.3f}（失效）")
    axes[2].scatter(X[:, 0], X[:, 1], c=db.labels_, cmap="coolwarm", s=16)
    axes[2].set_title(f"DBSCAN：ARI={ari_db:.3f}（成功）")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("独立研究②：k-均值假设簇是凸的球形，遇到环形数据必然切碎；DBSCAN 按密度连通可识别任意形状")
    fig.tight_layout()
    fig.savefig(OUT / "08_ring_dbscan.png", dpi=150)
    print(f"已写出：{OUT / '08_ring_dbscan.png'}")


def silhouette_experiment():
    df = pd.read_csv(DATA)
    feats = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
    X = StandardScaler().fit_transform(df[feats])

    ks = list(range(2, 10))
    scores, inertias = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        scores.append(silhouette_score(X, km.labels_))
        inertias.append(km.inertia_)
    print("\n真实数据各 k 的轮廓系数：")
    for k, s in zip(ks, scores):
        print(f"  k={k}: {s:.4f}")
    print(f"最高 {max(scores):.4f}（k={ks[int(np.argmax(scores))]}）；k=4 为 {scores[ks.index(4)]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar([str(k) for k in ks], scores, color="#2f6fed")
    bars[ks.index(4)].set_color("#e8590c")
    for i, s in enumerate(scores):
        ax.text(i, s, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("簇数 k")
    ax.set_ylabel("轮廓系数（-1~1，越大越紧凑分离）")
    ax.set_title("独立研究③：轮廓系数随 k 的变化（橙色为最终采用的 k=4）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "09_silhouette.png", dpi=150)
    print(f"已写出：{OUT / '09_silhouette.png'}")


if __name__ == "__main__":
    ring_experiment()
    silhouette_experiment()
