# Pilot 汇总报告 —— 两个真实 shard 的 Phase-1 全链路验证

日期: 2026-07-04 · 数据: Emilia-ZH `ZH-B000000` + `ZH-B000001` · 代码: `main@25e94c5`
（DNSMOS 移除 / dio@16k F0 / S2 后置 / Audiobox 分块 / CAM++ 跨 clip 批量 / bw≥4k / ce≥5.0）

## 1. 总漏斗

| 阶段 | 条数 | 占输入 | 时长 |
|---|---|---|---|
| 输入 | 68,792 | 100% | 169.9 h |
| S0 元数据预筛 | 51,241 | 74.5% | 121.3 h |
| S1 严格声学 | 15,823 | 23.0%（S0 的 30.9%） | — |
| S3 纯度合格 | **13,134** | **19.1%** | **28.6 h** |
| 发布集（再套 S2 top-40%） | **5,254** | 7.6% | ~11.5 h |

两 shard 高度一致（存活 19.8% vs 18.3%），阈值在同源数据上稳定。
按此比例外推 Emilia-ZH 全量（~920 shards / 5.4 万小时）：S3 合格集 ~9,200 小时，
发布集（top-40% 后）**~3,700 小时 / ~240 万条**。

## 2. 各闸门行为

**S0 拒绝原因**（17,551 拒）：`dnsmos_low` 15,156 · `duration_long` 1,864 · 复合 531

**S1 拒绝首因**（35,418 拒）：`aes_pq<7.0` 20,956 · `bandwidth<4k` 13,261 · `aes_ce<5.0` 1,029 · `snr<20dB` 171 · `clipping` 1

**S1 指标分位**（全体 S0 幸存者，p10 / p50 / p90）：

| 指标 | p10 | p50 | p90 | 闸门位置 |
|---|---|---|---|---|
| aes_pq | 6.38 | 7.12 | 7.56 | 7.0 ≈ p45（主刀）|
| aes_pc | 1.45 | 1.53 | 1.69 | 2.5 极松 |
| aes_ce | 4.74 | 5.39 | 5.88 | 5.0 ≈ p20 |
| snr_db | 28.5 | 44.2 | 64.0 | 20 极松 |
| bandwidth_hz | 2,188 | 3,906 | 7,688 | 4,000 ≈ p52 |

**S3 verdict**（15,823 条 S1 幸存者）：

| verdict | 条数 | 占比 | 处置 |
|---|---|---|---|
| single | 10,059 | 63.6% | 直接保留 |
| intruded_trimmed | 1,743 | 11.0% | 首/尾侵入已修剪回收 |
| degraded_pass | 1,332 | 8.4% | 保留（S5 发布闸门单独控）|
| intruded_rejected | 2,683 | 17.0% | 中段串音，拒 |
| overlap_rejected | 6 | 0.04% | 重叠说话，拒 |

**S2 韵律**（幸存者，p10/p50/p90）：f0_std 3.5/4.8/6.2 st · f0_range 11.0/15.2/20.7 st ·
语速 4.3/5.3/6.3 字/s · 停顿 0/1/4 次 · dio 置信度 0.57/0.68/0.79（`f0_confidence_poor=0.45` ≈ p3，符合"不误杀"定位）

**说话人**：存活集 995 人 / 13,134 条，均 13.2 条/人。

## 3. 性能（本次优化的实测结果）

| 项 | 优化前（估） | 优化后（实测） |
|---|---|---|
| 单 shard（85 h 音频）Phase-1 | 3–4 h | **~6 min（≈890× 实时）** |
| F0 | harvest 全量, 0.192x RT | dio@16k 仅幸存者, 0.008x RT |
| DNSMOS | 全量 CPU 串行 | 移除（边际拦截 ~2%）|
| Audiobox | 整 shard 单批 → OOM | 32 条/块 |
| CAM++ | 35.4 ms/窗（逐窗 pipeline）| ~11 ms/窗（跨 clip 512 窗批量, 余弦一致 1.000）|

全量外推：920 shards ≈ **~92 GPU·小时（单卡 4 天以内）**，与设计文档"2–4 天"目标一致，单卡即可。

## 4. 试听（12 条 top 优先级样本）

文件已导出到 `/opt/dlami/nvme/leolxliu/emilia-pilot/listen/`（`清单.txt` 含文本对照），
按 repack 优先级 `prosody_dsp_score × norm_pq` 排序：top-10 `single` + 2 条 `intruded_trimmed`（验证修剪效果）。
典型样本：pq 7.6–7.9 / ce 5.4–6.2 / f0_std 7–9 st（全体中位 4.8 → 选出来的确实"有戏"）。

## 5. HF 发布

- Repo: `leeoxiang/emilia-expressive-zh`（private）· 分支 `phase1-filtered`
- 内容: WebDataset shards（5,254 clips, trim 已应用）+ `metadata/phase1_metrics.parquet`（47 列全量指标, clip_id 主键）
- Phase-2 标签将以 `metadata/s4_labels.parquet` 增量追加，音频 tar 不重写

## 6. 遗留校准项

1. `aes_pq≥7.0` 在 p45 是有意为之的严档；若全量后想扩集，降到 6.9/6.8 即回放 SQL 一次的事；
2. `f0_confidence_poor=0.45` 在 dio 分布 p3，overlap_rejected 仅 6 条——若担心漏杀 overlap 可升到 0.55（p8）再看；
3. `intruded_rejected` 17% 偏高，值得抽 20 条人工听一下是真串音还是余弦阈值（0.70）过敏。
