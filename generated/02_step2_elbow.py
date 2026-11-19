# -*- coding: utf-8 -*-
"""第 9 次课 Step 2：肘部法则选 k（tag O9-step2）

k=1..9 逐个跑标准化后的 KMeans（random_state=42, n_init=10），记录 inertia_，
画"SSE 随 k 变化"折线并标注肘点；同时算轮廓系数作为第二个证据。

运行：python generated/02_step2_elbow.py
产出：outputs/02_elbow.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "bank_customers.csv"
OUT = BASE / "outputs"

FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
K_RANGE = range(1, 10)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    df = pd.read_csv(DATA)
    X = StandardScaler().fit_transform(df[FEATS])

    inertia, sil = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_) if k >= 2 else float("nan"))

    print("检查点 2：k 与 SSE / 轮廓系数")
    print(f"{'k':>2} {'inertia':>10} {'SSE 降幅':>10} {'轮廓系数':>10}")
    for i, k in enumerate(K_RANGE):
        drop = inertia[i - 1] - inertia[i] if i else float("nan")
        drop_s = f"{drop:.2f}" if i else "-"
        sil_s = f"{sil[i]:.4f}" if k >= 2 else "-"
        print(f"{k:>2} {inertia[i]:>10.2f} {drop_s:>10} {sil_s:>10}")

    k_best_sil = 2 + int(pd.Series(sil[1:]).idxmax())
    print(f"\n轮廓系数最高：k={k_best_sil}（{max(sil[1:]):.4f}）")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.plot(list(K_RANGE), inertia, "o-", color="#2f6fed", linewidth=2)
    ax.axvline(4, color="#f08c00", linestyle="--", linewidth=1.4)
    ax.annotate("肘点 k=4\n降幅由 131 收窄到 79", xy=(4, inertia[3]),
                xytext=(4.6, inertia[0] * 0.72),
                arrowprops=dict(arrowstyle="->", color="#f08c00"), color="#f08c00")
    ax.set_title("肘部法则：SSE（inertia）随 k 的变化")
    ax.set_xlabel("簇数 k")
    ax.set_ylabel("簇内平方和 SSE")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(list(K_RANGE)[1:], sil[1:], "s-", color="#2f9e44", linewidth=2)
    ax.axvline(4, color="#f08c00", linestyle="--", linewidth=1.4)
    ax.set_title("轮廓系数随 k 的变化（k≥2）")
    ax.set_xlabel("簇数 k")
    ax.set_ylabel("轮廓系数")
    ax.grid(alpha=0.3)

    fig.suptitle("选 k：肘部法拐点 k=4，轮廓系数峰值 k=%d（两者不一致，取业务可落地的 4）" % k_best_sil)
    fig.tight_layout()
    out = OUT / "02_elbow.png"
    fig.savefig(out, dpi=150)
    print(f"已写出：{out}")


if __name__ == "__main__":
    main()
