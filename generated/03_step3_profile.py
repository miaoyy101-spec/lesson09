# -*- coding: utf-8 -*-
"""第 9 次课 Step 3：分群画像与业务命名（tag O9-step3）

groupby('cluster') 输出每簇四特征均值，给每簇起中文业务名，画分组柱状图。

运行前提：先跑 01_step1_cluster.py
运行：python generated/03_step3_profile.py
产出：outputs/03_profile_table.csv、outputs/03_profile_bar.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"

FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
FEAT_CN = {
    "balance": "账户平均余额（元）",
    "avg_monthly_spend": "月均消费（元）",
    "product_count": "持有产品数（个）",
    "overdue_count": "逾期次数（次）",
}

# 四簇的中文业务命名：先看均值表，再看"这群人该用什么打法"
NAMES = {
    0: "高消费低黏性客群",
    1: "低频沉睡客群",
    2: "高余额高逾期风险客群",
    3: "多产品稳健主力客群",
}
SHORT = {0: "高消费低黏性", 1: "低频沉睡", 2: "高余额高逾期", 3: "多产品稳健"}
ADVICE = {
    0: "消费能力强但只持 2~3 个产品：主推交叉销售（信用卡分期、消费贷），把交易活跃转成产品黏性",
    1: "月均消费仅 7 千、产品少：低成本唤醒（活动券、App 推送），不投高成本人力",
    2: "余额最高但平均逾期 3 次：先做风险管控（额度复核、催收名单），再谈经营",
    3: "持 6~7 个产品、几乎不逾期：维护为主，专属权益 + 理财配置，防止流失",
}
COLORS = ["#2f6fed", "#868e96", "#e8590c", "#2f9e44"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    df = pd.read_csv(OUT / "bank_customers_clustered.csv")
    counts = df["cluster"].value_counts().sort_index()

    prof = df.groupby("cluster")[FEATS + ["age", "account_age_months"]].mean().round(2)
    prof.insert(0, "人数", counts)
    prof.insert(1, "占比", (counts / len(df) * 100).round(1))
    prof["业务命名"] = prof.index.map(NAMES)

    print("检查点 3：分群画像表")
    print(prof.to_string())
    prof.to_csv(OUT / "03_profile_table.csv", encoding="utf-8-sig")

    print("\n四簇业务命名与建议：")
    for cid in prof.index:
        print(f"  簇 {cid}｜{NAMES[cid]}（{counts[cid]} 人）：{ADVICE[cid]}")

    labels = [f"簇{c}\n{SHORT[c]}\n({counts[c]}人)" for c in prof.index]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.ravel(), FEATS):
        ax.bar(range(len(prof)), prof[feat], color=COLORS)
        ax.set_title(FEAT_CN[feat])
        ax.set_xticks(range(len(prof)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(prof[feat]):
            ax.text(i, v, f"{v:,.2f}" if v > 100 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("客户分群画像：四簇在四个特征上的均值（标准化后 KMeans，k=4，random_state=42）")
    fig.tight_layout()
    out = OUT / "03_profile_bar.png"
    fig.savefig(out, dpi=150)
    print(f"\n已写出：{out}")


if __name__ == "__main__":
    main()
