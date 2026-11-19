# -*- coding: utf-8 -*-
"""实践指南九·故意实验：不做标准化直接聚类

同一个 k=4、同一份数据、同一个 random_state，只差"有没有标准化"，
对比两种结果的画像表，看 balance 是否碾压其余三列。

运行：python generated/04_scale_experiment.py
产出：outputs/04_no_scale_compare.png
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
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def run(X, tag):
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    df = pd.read_csv(DATA)
    df["cluster"] = km.labels_
    counts = df["cluster"].value_counts().sort_index()
    prof = df.groupby("cluster")[FEATS].mean().round(1)
    prof.insert(0, "人数", counts)
    print(f"\n【{tag}】各簇人数：{counts.tolist()}｜inertia={km.inertia_:.2f}")
    print(prof.to_string())
    return df


def main():
    raw = pd.read_csv(DATA)
    df_raw = run(raw[FEATS].values, "不标准化")
    df_std = run(StandardScaler().fit_transform(raw[FEATS]), "标准化后")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, d, title in zip(
        axes,
        [df_raw, df_std],
        ["不标准化：分群被 balance 一列主导（按余额切四段）",
         "标准化后：四列等权参与距离（行为分群）"],
    ):
        sc = ax.scatter(d["balance"], d["avg_monthly_spend"],
                        c=d["cluster"], cmap="tab10", s=22, alpha=0.85)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("账户平均余额（元）")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("月均消费（元）")
    fig.suptitle("故意实验：不标准化 vs 标准化（k=4、random_state=42、四个特征均相同）")
    fig.tight_layout()
    out = OUT / "04_no_scale_compare.png"
    fig.savefig(out, dpi=150)
    print(f"\n已写出：{out}")


if __name__ == "__main__":
    main()
