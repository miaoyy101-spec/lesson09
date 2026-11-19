# -*- coding: utf-8 -*-
"""自主实践①：换特征集——加入 account_age_months 重新画像

对比"四特征（行为/资产）"与"五特征（+账户年限）"的分群结果，
回答：业务解读发生了什么变化？

运行：python generated/05_extra_feature.py
产出：outputs/05_extra_feature.png、outputs/05_extra_profile.csv
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

FEATS4 = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
FEATS5 = FEATS4 + ["account_age_months"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def cluster(df, feats):
    X = StandardScaler().fit_transform(df[feats])
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    return km.labels_, km.inertia_


def main():
    df = pd.read_csv(DATA)
    df["c4"], inertia4 = cluster(df, FEATS4)
    df["c5"], inertia5 = cluster(df, FEATS5)

    print("四特征各簇人数：", df["c4"].value_counts().sort_index().tolist(),
          f"｜inertia={inertia4:.2f}")
    print("五特征各簇人数：", df["c5"].value_counts().sort_index().tolist(),
          f"｜inertia={inertia5:.2f}")
    ari = adjusted_rand_score(df["c4"], df["c5"])
    print(f"两种分群的调整兰德指数 ARI = {ari:.3f}（1=完全一致，0=随机）")

    prof = df.groupby("c5")[FEATS5].mean().round(1)
    prof.insert(0, "人数", df["c5"].value_counts().sort_index())
    print("\n五特征分群画像（重点看 account_age_months）：")
    print(prof.to_string())
    prof.to_csv(OUT / "05_extra_profile.csv", encoding="utf-8-sig")

    print("\n簇编号迁移（行=四特征簇，列=五特征簇）：")
    print(pd.crosstab(df["c4"], df["c5"]).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    cnt4 = df["c4"].value_counts().sort_index()
    cnt5 = df["c5"].value_counts().sort_index()
    axes[0].bar(cnt4.index.astype(str), cnt4.values, color="#2f6fed")
    axes[0].set_title("四特征（行为/资产）各簇人数")
    axes[0].set_ylabel("人数")
    for i, v in enumerate(cnt4.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    axes[1].bar(cnt5.index.astype(str), cnt5.values, color="#e8590c")
    axes[1].set_title("加入 account_age_months 后各簇人数")
    axes[1].set_ylabel("人数")
    for i, v in enumerate(cnt5.values):
        axes[1].text(i, v, str(v), ha="center", va="bottom")
    fig.suptitle(f"自主实践①：换特征集前后的人数结构（ARI={ari:.3f}）")
    fig.tight_layout()
    out = OUT / "05_extra_feature.png"
    fig.savefig(out, dpi=150)
    print(f"\n已写出：{out}")


if __name__ == "__main__":
    main()
