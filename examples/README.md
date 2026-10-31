# 样例轨迹格式

`commit_trace.sample.json` 是 1 条真实验证 episode（`00746-RTV2n6fXB2w_07113`，17 个事件）的回放轨迹，
由 `python -m m3assign.commit_trace` 导出，只包含回放需要的字段，不含原始图像与数据集路径。

## 顶层

| 字段 | 说明 |
|---|---|
| `ckpt` | 生成轨迹时用的 checkpoint（仅作标识） |
| `margin` | 离线评测用的余量 λ（回放页的默认值） |
| `variants` | 三种解码变体名：`per_frame`、`fresh@<λ>`、`memory@0` |
| `episodes` | episode 列表，每条含 `id`、`K`、`M`、`oracle`、`corr`、`harm`、`events` |

## episode

| 字段 | 说明 |
|---|---|
| `id` | episode_id |
| `K` / `M` | 当前 episode 的 agent 数 / 目标数（≤ 4 / 5） |
| `oracle` | 长度 M，每个目标在真值最优分配里分给哪个 agent（-1 表示无） |
| `corr` / `harm` | 该 episode 内 `fresh@λ` 的纠正 / 有害切换计数 |
| `events` | 按时间排序的事件 |

## event

| 字段 | 说明 |
|---|---|
| `t` | 事件序号（教师快照的 event_id） |
| `reason` | `start` / `periodic` / `found` |
| `map` | BEV 的 96×96 PNG，data URL（R=占据/可见度，G=已探索） |
| `agents` | K 个 `[x, y]`，BEV 96×96 地图像素坐标（0–96，可带小数） |
| `goals` | M 个目标 `[x, y]`，同上 |
| `conf` | 每个目标的置信度 |
| `valid_k` / `valid_m` | agent / 目标是否有效 |
| `teacher` | 教师给每个 agent 指派的目标（-1 表示无） |
| `oracle` | 该帧真值最优分配（长度 M） |
| `vars` | 变体名 → 每个 agent 的一行决策 |
| `logits` | K×M 的 contract-free 打分，回放页据此对任意 λ 重新解码 |

## 决策行（`vars[变体][a]`）

| 字段 | 说明 |
|---|---|
| `a` | agent 下标 |
| `old` / `new` | 上一步 / 本步的目标（-1 表示无） |
| `act` | `keep` / `corr` / `harm` / `neutral` / `forced` / `absent`；`forced` 且 `old < 0` 时页面显示为“初始” |
| `gap` | 保持时 = 最佳替代 − 当前目标；切换时 = 新目标 − 旧目标 |

## 重建回放

```bash
python -m m3assign.commit_view --trace examples/commit_trace.sample.json --out /tmp/replay.html
```

浏览器端与离线评测的差异（并列解）见根目录 README 的“已知限制”。
