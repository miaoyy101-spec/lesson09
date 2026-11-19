# -*- coding: utf-8 -*-
"""自主实践②：簇的稳定性——random_state 换三个值，画像是否大变？

固定 k=4、四特征、标准化不变，只换 random_state（42 为基准，另取 0、7、2024），
比较各簇人数、inertia 与调整兰德指数 ARI。

运行：python generated/06_stability.py
产出：outputs/06_stability.png、outputs/06_stability.csv
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "bank_customers.csv"
OUT = BASE / "outputs"

FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
SEEDS = [42, 0, 7, 2024]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    df = pd.read_csv(DATA)
    X = StandardScaler().fit_transform(df[FEATS])

    rows, labels = [], {}
    for seed in SEEDS:
        km = KMeans(n_clusters=4, random_state=seed, n_init=10).fit(X)
        labels[seed] = km.labels_
        cnt = pd.Series(km.labels_).value_counts().sort_index().tolist()
        rows.append({
            "random_state": seed,
            "各簇人数": "/".join(map(str, cnt)),
            "inertia": round(km.inertia_, 2),
            "ARI(vs 42)": 1.0 if seed == 42 else round(adjusted_rand_score(labels[42], km.labels_), 3),
        })

    report = pd.DataFrame(rows)
    print("自主实践②：不同随机种子的稳定性")
    print(report.to_string(index=False))
    report.to_csv(OUT / "06_stability.csv", index=False, encoding="utf-8-sig")

    # 画像稳定性：每个种子下"月均消费最高的一簇"与"最低的一簇"的均值
    print("\n画像是否大变（用月均消费和逾期次数两个关键列看结构）：")
    for seed in SEEDS:
        d = df.copy()
        d["c"] = labels[seed]
        g = d.groupby("c")[["avg_monthly_spend", "overdue_count"]].mean().round(0)
        print(f"  seed={seed}：月均消费 {g['avg_monthly_spend'].min():.0f}~{g['avg_monthly_spend'].max():.0f} 元，"
              f"逾期次数 {g['overdue_count'].min():.2f}~{g['overdue_count'].max():.2f} 次")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.2
    for i, seed in enumerate(SEEDS):
        # 簇号只是编号，按人数从大到小排序后再比，才能看出"结构"是否稳定
        sizes = sorted(pd.Series(labels[seed]).value_counts().tolist(), reverse=True)
        ax.bar([x + i * width for x in range(4)], sizes, width=width, label=f"seed={seed}")
        for x, v in enumerate(sizes):
            ax.text(x + i * width, v, str(v), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks([x + 1.5 * width for x in range(4)])
    ax.set_xticklabels(["第1大簇", "第2大簇", "第3大簇", "第4大簇"])
    ax.set_ylabel("人数")
    ax.set_ylim(0, 85)
    ax.set_title("自主实践②：四个随机种子下按规模排序的各簇人数（结构性稳定，簇号会轮换）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / "06_stability.png"
    fig.savefig(out, dpi=150)
    print(f"\n已写出：{out}")


if __name__ == "__main__":
    main()
