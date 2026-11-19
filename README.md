# lesson09-cluster

《人工智能程序设计与实践》2026 秋 · 第 9 次课课堂作业：**聚类与客户分群**（实践指南九）

- 数据：`data/bank_customers.csv`（200 位客户 × 12 列，原文件不改动，仅读取）
- 聚类特征：`balance`、`avg_monthly_spend`、`product_count`、`overdue_count`
  （选行为/资产特征而不是 age：age 会把分群变成"年龄分组"，`is_default` / `risk_level` 是标签列，无监督不能用；`customer_id` 是编号不进模型）
- 结论速览：标准化后 KMeans(k=4, random_state=42, n_init=10) → **四簇 52 / 58 / 20 / 70 人**

## 目录结构

```
lesson09-cluster/
├─ data/bank_customers.csv            # 课程原始数据（未改动）
├─ generated/
│  ├─ 01_step1_cluster.py             # Step1 聚类跑通（tag O9-step1）
│  ├─ 02_step2_elbow.py               # Step2 肘部法则选 k（tag O9-step2）
│  ├─ 03_step3_profile.py             # Step3 画像 + 四中文命名（tag O9-step3）
│  ├─ 04_scale_experiment.py          # 指南"故意实验"：不标准化直接聚类
│  ├─ 05_extra_feature.py             # 自主① 加入 account_age_months 重画像
│  ├─ 06_stability.py                 # 自主② 三种子稳定性
│  ├─ 07_scatter_2d.py                # 自主③ scatter 二维投影
│  └─ 08_independent_study.py         # 独立研究②环形+DBSCAN、③轮廓系数
├─ outputs/                           # 全部图表与表（脚本生成，可复现）
├─ prompts/ai-log.md                  # AI 使用记录：指令原文 + 验证过程
├─ progress.md                        # 进度报告（提交用）
└─ independent-study.md               # 独立研究任务书面回答
```

## 运行方式

```bash
pip install pandas numpy scikit-learn matplotlib
python generated/01_step1_cluster.py
python generated/02_step2_elbow.py
python generated/03_step3_profile.py
python generated/04_scale_experiment.py
python generated/05_extra_feature.py
python generated/06_stability.py
python generated/07_scatter_2d.py
python generated/08_independent_study.py
```

## 检查点结果

### 检查点 1：四簇人数（`outputs/bank_customers_clustered.csv`）

| 簇 | 人数 | 占比 |
|---|---|---|
| 0 | 52 | 26.0% |
| 1 | 58 | 29.0% |
| 2 | 20 | 10.0% |
| 3 | 70 | 35.0% |

inertia（SSE）= 434.33，轮廓系数（k=4）= 0.2275

### 检查点 2：肘部图（`outputs/02_elbow.png`）

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| SSE | 800.00 | 644.69 | 513.81 | **434.33** | 366.21 | 317.67 | 280.11 | 257.21 | 236.66 |
| 降幅 | - | 155.31 | 130.88 | **79.48** | 68.12 | 48.54 | 37.56 | 22.91 | 20.55 |
| 轮廓系数 | - | 0.1987 | 0.2281 | **0.2275** | 0.2405 | 0.2491 | 0.2642 | 0.2612 | 0.2591 |

选 k=4 的一句话论证：SSE 降幅在 k=3→4 由 130.88 腰斩到 79.48（肘点），而轮廓系数虽在 k=7 最高（0.2642）但只比 k=4 高 0.037、且全部取值都低于 0.3（结构本身不强），k>4 拆出的群运营上也无法各配一套打法，故取 k=4。

### 检查点 3：画像表 + 四个中文业务命名（`outputs/03_profile_table.csv`、`outputs/03_profile_bar.png`）

| 簇 | 人数 | 平均余额(元) | 月均消费(元) | 产品数 | 逾期次数 | 业务命名 |
|---|---|---|---|---|---|---|
| 0 | 52 | 238,836 | 23,983 | 2.81 | 0.46 | **高消费低黏性客群** |
| 1 | 58 | 243,864 | 6,985 | 2.91 | 0.41 | **低频沉睡客群** |
| 2 | 20 | 298,909 | 16,234 | 5.45 | 3.05 | **高余额高逾期风险客群** |
| 3 | 70 | 248,965 | 15,928 | 6.67 | 0.14 | **多产品稳健主力客群** |

业务建议：簇 0 交叉销售提黏性；簇 1 低成本唤醒；簇 2 先风控后经营；簇 3 权益维护防流失。

## 自主实践与独立研究结果（详见 `progress.md`）

- 加入 `account_age_months` 后人数变为 62/49/64/25，与四特征方案 ARI 仅 0.293 —— 客龄主导了距离，分群从"行为价值"变成"客龄分层"。
- 三种子（0/7/2024）：按规模排序后仍是 ≈70/58/52/20，结构稳定；但个体归属 ARI 只有 0.45~0.97，说明簇边界模糊，只能做群体策略、不宜按簇号做个体精准动作。
- 环形数据：k-均值 ARI = -0.002（切碎两个环），DBSCAN ARI = 1.000。
- 轮廓系数：k=2..9 全部在 0.20~0.26，印证"分群是业务决策，不是数学唯一解"。

## Git 提交规范

- `O9-step1` / `O9-step2` / `O9-step3` 三个 tag 对应三步课堂任务
- 之后按"自主实践 / 独立研究 / 进度报告"分别提交
