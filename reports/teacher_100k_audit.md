# HM3D-M3 官方 `teacher_100k` 样本审计 + 承诺字段实验

生成脚本：[m3assign/audit_data.py](../m3assign/audit_data.py)、
[m3assign/churn.py](../m3assign/churn.py)
机器可读结果：`cache/audit_teacher.json.gz`、`cache/churn.json.gz`、
`cache/sim_eval_pair_couple.json`、`cache/sim_eval_pair_contract3.json`
（定版评测 `cache/sim_eval.json` 未被改动，md5 `36fdbd12098284d7301653a8c5e9dee7`）

复现：

```bash
cd m3-assign   # 仓库根目录
python -m m3assign.audit_data --all --geom 1500   # ~46 s，流式扫 98 个 shard
python -m m3assign.churn                          # ~7 s，只读小标签张量
python -m m3assign.train --train-shards 40 --val-shards 4 --epochs 40 --batch 512 \
    --lr 6e-4 --couple-iters 3 --contract 1 --name contract3     # 187 s / RTX 3060 Ti
python -m m3assign.sim_eval --episodes 400 --detail 400 --ckpt runs/<run>/best.pt \
    --out cache/sim_eval_pair_<tag>.json
```

`--geom N` 用 habitat navmesh 把这 N 个 episode 的真测地与 `travel` 做秩相关；其余统计不需要 habitat。

## 1. 覆盖范围

| 项 | 值 |
|---|---|
| 行（快照帧） | 100 000（98 shard × 1024） |
| episode | 15 917 |
| `reason=start` | 12 251（12.3%） |
| `reason=periodic` | 53 723（53.7%） |
| `reason=found` | 34 026（34.0%） |

每 episode 行数中位 5、p90 14；84 083 次帧间转移里 68 157 次（81%）是相邻事件（`event_id` 差 1）。

## 2. 坐标系：SCHEMA 写的 `[-1,1]` 与数据不符，我们的实现是对的

文档声称 `pose`/`peak` 归一化到 `[-1,1]`。实测 `pose` 值域 `[-2.0, 2.0]`，
**278 331 / 295 836（94.1%）** 个 agent 坐标越出 `[-1,1]`；`peak` 越界 241 915 / 488 864（49.5%）。
真实帧是**以 −1 为中心的 `[-2,0]`**（`grid_sample` 前需 `+1.0`，翻转反射是 `-x-2`）。

配准测试（每 shard 抽 120 行，投到 96×96 单元格，量到"应该落在哪"的距离）：

| 判据 | 我们的 `+1` 帧 | 随机基线 |
|---|---|---|
| agent 到已探索地形（`geo` ch0 非零格）中位距离 | **1.58 格** | 18.03 格 |
| 落在 5 格以内的比例 | **68.1%** | 22.0% |
| goal 到该目标热图团的中位距离 | **2.79 格** | 21.93 格 |
| 落在 5 格以内的比例 | **58.2%** | 16.0% |

轴向约定用"同一行内 pose 分量与行/列索引的秩相关"判定（564 行）：
`x~col` 中位 ρ = **1.000**（98.8% 为正）、`y~row` ρ = **1.000**（99.3%），交叉项 ρ ≈ 0、正负各半
⇒ **x=列、y=行**，与 `model.py` 的 `grid_sample` 用法一致。

**结论：这是官方文档的 bug，不是我们的 bug。** `model.py:80-82` 与 `train.py:_flip` 保持不变；
`[-2,2]`（除以 2）的解释处在随机水平，可排除。

## 3. `travel` 字段基本没有信息量

`travel` 被描述为 agent→goal 的（测地）代价，也是逐对特征里最"应该有用"的一个。按 `reason` 分组：

| reason | 有效对 =0 | =1e4(不可达) | teacher 所在对 =0 | 未分配对 =0 | ρ(travel, 欧氏) | ρ(travel, 真测地) |
|---|---|---|---|---|---|---|
| start | 71.9% | 14.4% | 72.0% | 71.7% | 0.237 | **0.051** |
| periodic | 65.1% | 16.3% | 63.7% | 66.7% | 0.284 | – |
| found | 63.8% | 17.3% | 62.2% | 65.1% | 0.291 | – |

1. **63–72% 的"代价"恰好等于 0**，另 14–17% 是哨兵值 1e4，剩下的才是真数。
2. 与 habitat navmesh 真测地（start 帧，1500 个 episode）秩相关 **0.051** ≈ 无关；与平面欧氏也只有 0.24–0.29。
3. "teacher 把 agent 分给它时 travel 更可能为 0"不成立：start 帧 72.0% vs 71.7%，差 0.3 个百分点。
   ⇒ `travel=0` **不是**"该 agent 已在此目标上"的标记，而是**缺测**。

这解释了此前的对照实验：把 `travel` 整体换成欧氏距离，MSPL 只动 **+0.0097（p=0.25）**；
而用真测地跑 Hungarian 比模型高 **+0.0652（p<1e-4）**。缺的是**拓扑信息**，不是解码器。

## 4. 空间通道极稀疏；`peak_u` 不总是热图峰值

| 通道 | 非零格占比 |
|---|---|
| `geo` ch0（已探索） | 5.21%（start 帧仅 3.65%） |
| `geo` ch1 | 10.78% |
| `heat` | 6.95%（start 帧 4.11%） |

- `geo` ch0 在同一 episode 内有 **9004 / 84004 行完全恒定**：所谓"探索进度"在很多 episode 里是静态底图。
- `peak[...,2]`（文档说是该目标热图峰值强度）与热图实际最大值相符的只有
  **150 498 / 227 623（66%）**，中位 u 仅 0.28–0.36。

对模型的直接影响：start 帧上网络能拿到的地图信息只有 3.7% 的已探索格 + 4.1% 的热图格，
`travel` 又大面积缺测 —— start 帧的分配任务在这些输入下天花板本来就低。

## 5. 标签结构：合法，但 `teacher` 给闲置 agent 补了值

| 检查 | 结果 | 判定 |
|---|---|---|
| `valid_k != K` | 0 | 正常 |
| `teacher != teacher_gt` | 0 | 两个字段同义 |
| `teacher` 指向无效目标 | 0 | 标签合法 |
| start 帧 `valid_m != M` | 0 | 起始时目标全部有效，符合设计 |
| start 帧有效 agent 上 `teacher == -1` | 0 | 起始不遗漏 agent |
| `teacher != teacher_core` | **156 472 格** | `teacher` 给 idle agent 也填了值，`teacher_core` 才是无冲突子集 |
| `valid_m != M`（全部帧） | 78 060 行 | 目标被解决后掉出可用集，属设计 |

**口径提醒（很容易把两个数字混着说）：**
`metrics()["acc"]` 是逐对 argmax 的复现率；`teacher_acc` 是先做同信息 Hungarian 一对一解码再比教师标签，
因为教师标签本身带冲突（上一行），解码会"纠正"掉一部分，所以 `teacher_acc < acc` 是正常现象：
定版 couple3r 为 `acc=0.578`、`teacher_acc=0.332`、`cf_acc=0.329`。
下文的单帧基线全部与 **argmax 口径（`acc`）** 可比，不要跟 `teacher_acc` 比。

## 6. `contract` 字段：部署时可得的自身状态，不是同帧泄漏

`contract`（本帧之前每个 agent 承诺在哪个目标）此前谁都没喂给网络。
先把它的因果方向钉死（`churn.contract_semantics`，100 000 行全量）：

| 比较 | start | periodic | found | 合计 |
|---|---|---|---|---|
| `contract[t] == teacher[t]`（同帧复读 = 泄漏） | 0.000 | 0.9895 | 0.4297 | 0.674 |
| `contract[t] == teacher[t-1]`（上一帧决策 = 记忆） | – | **0.9481** | **0.9476** | **0.948** |
| `contract[t] == contract[t-1]` | – | 0.3138* | 0.6821* | 0.456* |

（*按 `teacher_core` 口径，见 `contract_eq_prev_core`。）

- start 帧 `contract` **恒为 −1**（`contract_ne_minus1_at_start = 0`），所以同帧比较在 start 上是 0 —— 它不可能是标签回填。
- `contract[t]` 有 94.8% 等于**上一帧的 teacher 决策**，两个非起始 reason 上都是 0.948 ——
  它就是"上一帧下达的承诺被带进这一帧"。
- ⇒ **把它当输入是合法的**：部署时 agent 本来就知道自己当前的承诺。

但它同时也说明：教师标签高度自相关，"复读承诺"不是更聪明的分配，而是**更少改判**。

## 7. 教师是惯性复读：跨事件动力学

68 157 次相邻事件转移、201 251 个 `teacher_core` 格：

| 指标 | 值 |
|---|---|
| 标签变化率 | **19.4%** |
| 其中"目标被解决，不得不改"（forced） | 83.2% |
| 真正自由的改判（free churn） | **16.8%**（6 546 格，只占全部格子的 3.3%） |
| 上一帧的无冲突承诺本帧仍是有效目标时（71 714 格），教师保留它的比例 | **90.9%** |
| 直接复读上一帧 `teacher_core` 的准确率 | **80.6%** |

（此前流传的"oracle 只保留 55.3%/56.8% 的有效承诺"是脚本 bug：
`prev core` 与 `own contract` 两个来源共用了一个计数器，分母翻倍。已拆成
`oracle_keeps_prev_core_rate` / `oracle_keeps_own_contract_rate` 两个字段。
按无冲突承诺口径，教师其实是**高度保守**的：91% 保留，自由改判只占 3.3% 的格子。）

单帧基线（模型自己的分母，295 836 格，argmax 口径）：

| 策略 | 复现率 |
|---|---|
| **只抄本行 `contract`** | **0.674** |
| greedy：`travel` 最小 | 0.571 |
| greedy：欧氏最近 | 0.563 |
| 抄 `teacher_core` | 0.471 |
| 多数类先验（目标频率 `{0:79k,1:76k,2:74k,3:46k,4:21k}`） | 0.267 |
| 定版模型 couple3r（`acc`，val） | 0.578 |

也就是说：**定版模型复现教师的比例低于"直接抄承诺"这个零参数策略**。
这是数据性质决定的：periodic 帧（53.7% 的行）上教师根本不重新决策，`contract == teacher` 的比例是 **98.95%**；
只有 `found` 帧（34.0% 的行）才真的需要重新分配，那里 `contract` 只剩 43.0%。

## 8. 实验：把 `contract` 喂进 AssignNet（`contract3`）

改动很小：`use_contract=True` 时逐对特征从 6 维加到 9 维
（`own`＝本 agent 承诺在该目标、`taken`＝他人承诺数、`has`＝本 agent 有承诺），
`pair_f`、`build()`、`--contract` 三处跟着走；`_flip` 不动 contract（目标索引与翻转无关）。

同配置对照（40 训练 shard / 4 验证 shard 94–97 / 40 epoch / batch 512 / lr 6e-4 / couple_iters 3，均 187 s）：

| 指标 | couple3r（定版） | contract3 |
|---|---|---|
| val loss | 0.751 | **0.395** |
| val `acc`（argmax 复现率） | 0.578 | **0.847** |
| val `teacher_acc`（Hungarian 解码后） | 0.332 | **0.751** |
| val `cf_acc`（无冲突子集） | 0.329 | **0.698** |

按 reason 拆开（val 全量 3 744 行，argmax 口径，同一脚本 `/tmp/perreason.py`）：

| reason | 抄 contract | couple3r | contract3 |
|---|---|---|---|
| start（n=1251） | – | 0.281 | 0.282 |
| periodic（n=6682） | 0.992 | 0.637 | **0.992** |
| found（n=3308） | 0.452 | 0.624 | **0.772** |
| 合计 | 0.674 | 0.593 | **0.848** |

- periodic 帧被 contract 直接钉死（0.992 = 复读上限），说明教师在那里确实不做决策。
- **`found` 帧是唯一需要真分配的子集**：contract3 0.772，比复读基线 0.452 高 **32 个百分点**，
  也比定版模型高 14.8 个百分点 —— 这部分才是网络学到的东西。
- start 帧纹丝不动（0.281 → 0.282）：`contract` 在该帧恒为 −1，三个新特征全零，网络退化回 6 特征模型。

同一批 400 episode 的起点帧部署评测（navmesh 真测地，配对符号检验 + bootstrap，
`assignnet_hung` = 模型分数 → 一对一 Hungarian）：

| 指标 | couple3r | contract3 | Δ | 95% CI | p |
|---|---|---|---|---|---|
| MSPL | 0.7765 | 0.7894 | +0.0129 | [−0.0039, +0.0298] | 0.14 |
| regret | 1.4148 | 1.3939 | −0.0208 | [−0.0603, +0.0185] | 0.31 |
| makespan | 17.353 | 17.266 | −0.088 | [−0.598, +0.399] | 0.74 |
| agree_oracle | 0.4160 | 0.4534 | **+0.0375** | [+0.0054, +0.0693] | **0.024** |

起点帧是 harness 唯一能定价的帧，而那里 contract 恰好没有信息 —— 所以路线质量类指标全部不显著
（方向都对模型有利：regret/makespan 略降），唯一显著的是"与 oracle 分配一致率"+3.75 点。

## 9. 这次审计改变了什么判断

1. **坐标系实现无误**：`[-2,0]` 帧、x=列 y=行由配准测试独立证实；此前对 `peak` 配准的怀疑是小样本噪声。
2. **`travel` 不能当代价特征用**：ρ(真测地)=0.05、2/3 取值为 0。凡基于它的 greedy/"oracle 已在目标上"推断都要打折。
3. **start 帧信息天花板极低**：已探索 3.7% + 热图 4.1% + travel 缺测 + contract 恒为 −1。
   看板评测取的正是这个最难的子集（406/513 行为 start）。
4. **教师行为本身是个可研究对象**：83% 的改判由目标解决触发，自由改判只占全部格子的 3.3%，
   上一承诺仍有效时教师 90.9% 选择保留；53.7% 的行（periodic）根本不重新决策。
   论文把这批数据当"最优分配示范"，审计显示它是**以保留承诺为默认动作、只在 found 事件上重决策**的在线策略。
5. **`contract` 是释放数据里唯一携带真信号的时序字段**，且合法可用（它是上一帧决策，94.8%；同帧泄漏为 0）。
   喂进网络后 val 复现率 0.578→0.847、无冲突子集 0.329→0.698，且在唯一需要决策的 `found` 帧
   高出复读基线 32 点。但它**不改善**起点帧的路线质量（不显著）。
   因此 headline 不能写"更会规划"，要写"承诺感知 + 重决策帧准确率"；
   真正的规划增益还得等图级序贯 rollout / decision-focused 训练（方向 1）。
