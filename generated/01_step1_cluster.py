# -*- coding: utf-8 -*-
"""第 9 次课 Step 1：聚类跑通（tag O9-step1）

对 bank_customers.csv 选取 balance, avg_monthly_spend, product_count, overdue_count
四列标准化后做 KMeans(k=4)，把簇编号写回 df 新列 cluster，输出各簇人数。

运行：python generated/01_step1_cluster.py
产出：outputs/bank_customers_clustered.csv（原数据 + cluster 列，原文件不改动）
"""
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "bank_customers.csv"
OUT = BASE / "outputs"

# 为什么用这四列：它们是"行为/资产"特征，能刻画客户价值与风险；
# customer_id 是编号、age 会把分群变成"年龄分组"、is_default/risk_level 是标签列（无监督不能用）。
FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
K = 4


def main():
    df = pd.read_csv(DATA)
    print(f"数据规模：{df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"参与聚类的特征：{FEATS}")

    # 1) 标准化：四列量纲差距大（balance 十万级、overdue_count 个位数），
    #    不标准化会被 balance 一列主导，分群退化成"按余额切四段"。
    X = StandardScaler().fit_transform(df[FEATS])

    # 2) 聚类：n_init=10 显式写出，保证不同机器/版本跑出一致结果
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X)

    # 3) 各簇人数
    counts = df["cluster"].value_counts().sort_index()
    print("\n检查点 1：四簇人数")
    for cid, n in counts.items():
        print(f"  簇 {cid}：{n} 人（{n / len(df):.1%}）")
    print(f"合计：{len(df)} 人")
    print(f"inertia（SSE）：{km.inertia_:.2f}")
    print(f"轮廓系数（k=4）：{silhouette_score(X, df['cluster']):.4f}")

    # 4) 簇编号写回并落盘（原始 csv 保持不动，满足"数据文件不得改动"）
    out_csv = OUT / "bank_customers_clustered.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n已写出：{out_csv}")


if __name__ == "__main__":
    main()
