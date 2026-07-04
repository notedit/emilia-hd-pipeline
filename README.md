# Emilia 高表现力子集抽取 Pipeline

从 Emilia-ZH（~5 万小时）中筛出**声学干净、韵律动态丰富、情感表达充分**的高质量子集，
为每条数据产出结构化标签，分层（Tier-S/A/B）发布到 HuggingFace，供情感 TTS 训练与
prosody 后训练使用。

核心原则：**漏斗式级联（便宜的在前）· 数值全部入库、判定只是查询条件 · 纯文件存储
（parquet + npy + done marker）· anytime 可中断**。详细设计见
[emilia_expressive_pipeline_design.md](emilia_expressive_pipeline_design.md)，
真跑手册见 [docs/PILOT_RUNBOOK.md](docs/PILOT_RUNBOOK.md)。

## 总体流程

```
Phase 1（全量扫描, 每 tar shard 一个任务, GPU×N + CPU 池）
  S0 元数据预筛 → CPU pass A(解码+S1 CPU 指标) → S1 声学闸门(GPU)
      → CPU pass B(S2 韵律特征, 仅 S1 幸存者) → S3 滑窗声纹纯度(GPU)
      → 各 stage parquet + emb npy + done marker（shard 级原子落盘）
  ──► 重打包 repack: 幸存 clip 按打标优先级写新 WebDataset + worklist
  ──► (可选短路) Phase-1 过滤子集直接发布 HF `phase1-filtered` 分支

Phase 2（云端 API 打标, worklist 分片认领, anytime 可停）
  S4 Qwen3-Omni 结构化打标 → s4_labels parquet → 增量上传 HF metadata

收尾
  S5 综合评分 + 分 Tier（DuckDB, 分钟级）──► HF 完整版发布（main 分支）
```

## 各 Stage 职责

### S0 · 元数据预筛（`stages/s0_prefilter.py`，纯 metadata，不解码音频）

只读 Emilia 自带 JSON 字段，产出每 shard 的白名单（含拒绝原因）：

| 条件 | 阈值 | 理由 |
|---|---|---|
| duration | 3.0 ≤ d ≤ 20.0 s | 短于 3s 无韵律结构，长于 20s 情感不纯 |
| language | == zh | 本期只做中文 |
| 原始 dnsmos | ≥ 3.2 | 比 Emilia 原始 3.0 收紧的粗刀 |
| text | 非空且 ≥ 4 字 | 空文本无法做校验 |

输出 `stage/s0_prefilter/part-{shard}.parquet`（**所有** clip 一行，不丢行）。

### S1 · 严格声学过滤（`stages/s1_acoustics.py`，GPU×2 + CPU DSP）

| 指标 | 模型/方法 | 阈值（初值，待校准） |
|---|---|---|
| aes_pq / aes_pc | Audiobox-Aesthetics（GPU，32 条/块前向） | pq ≥ 7.0（主闸门）· pc ≤ 2.5 |
| aes_ce | 同上 | ≥ 5.0（content enjoyment 下限，~p20） |
| aes_cu | 同上 | 不设闸门，入库供 S5 |
| snr_db | 能量分位法（CPU） | ≥ 20 dB |
| clipping_ratio | CPU | ≤ 0.001 |
| bandwidth_hz | 频谱 roll-off（CPU，防上采样假高清） | ≥ 8 kHz |
| loudness_lufs | pyloudnorm（CPU） | 入库不设闸门 |

**DNSMOS 已从 S1 移除**（onnx 只能 CPU 串行且慢；实测在 aes_pq≥7.0 之上的边际拦截率
仅 ~2%）——S0 保留 Emilia 自带 `dnsmos ≥ 3.2` 的 metadata 粗筛，行 schema 中 `dnsmos_*`
列保留（NULL）以兼容旧数据。pass/reject 仅是建议列，改阈值 = 改 SQL 重放，不重跑管线。

### S2 · 韵律丰富度粗筛（`stages/s2_prosody.py`，CPU DSP，**仅 S1 幸存者**）

判"有没有戏"，砍平淡朗读，故意宽松（精判留给 S4）：

| 指标 | 说明 |
|---|---|
| f0_std_st / f0_range_st | log-F0 半音域标准差 / P5-P95 范围（跨性别可比）。F0 用 **pyworld dio+stonemask @16k**（比 harvest 快 ~24×） |
| energy_std_db | 帧能量标准差 |
| speech_rate_cps / rate_var_cv | 字/秒（字数来自 Emilia text）及滑窗语速变异系数 |
| pause_count / pause_total_ms | silero-VAD 停顿统计 |
| f0_tracker_confidence | voiced 占比 × (1−跳变率)，**透传给 S3 做 overlap 旁证** |

`prosody_dsp_score`（z-score 加权和）**不在本级落盘**——它必须在全体存活样本上归一化，
由 DuckDB 在 repack/S5 时全局计算；top-40% 分位闸门同样后置。

### S3 · 滑窗声纹纯度（`stages/s3_speaker.py`，CAM++，无全局聚类）

只回答一个问题：**这条 clip 内部是不是单说话人、无串音/overlap**。说话人身份直接沿用
Emilia 的 `original_speaker`，不做跨 clip 归并。

1.5s 窗、50% overlap 提取 CAM++ 窗级 embedding，用窗到均值中心的余弦几何判决：

| 余弦形态 | F0 置信度 | verdict |
|---|---|---|
| 局部连续塌陷（可定位） | — | 首/尾侵入→**intruded_trimmed**（修剪回收，余量≥3s）；中段→intruded_rejected |
| 均匀压低 | 差 | overlap_rejected（疑似重叠说话） |
| 均匀压低 | 正常 | degraded_pass（放行，交给分数体系兜底） |
| 正常 | — | single |

输出 features parquet + clip 级均值 embedding（`emb-{shard}.npy`, fp16, 192 维）。
判决偏向不误杀；被拒的行也入库。

### 重打包（`phase1/repack.py`，Phase-1→Phase-2 桥）

DuckDB join S0-S3 全部 parquet → 幸存者（S0/S1 passed + verdict 合格 + S2 top-40%）
按 `priority = prosody_dsp_score × norm_aesthetics_pq` 降序 → 写
`manifests/s4_worklist_v1.parquet`（默认 5000 clip/分片）+ 按优先级排序的新 WebDataset
shard（`intruded_trimmed` 的音频在此真正修剪）。**最有戏的先打标**，Phase 2 随时停，
已完成部分即当前最优子集。

### S4 · Qwen3-Omni 结构化打标（`phase2/s4_client.py` + `dispatch.py`，云端 API）

只做必须"听懂内容"的任务：情感（主/次 + 强度）、韵律/表现力、语境（朗读/对话/演播…）、
文本校验与标点修复、语种确认。asyncio 并发 + 指数退避重试 + prompt 强制 JSON +
客户端 schema 校验（`S4GuidedJSON`）。任务粒度 = worklist 分片，done marker 断点续跑。
输出 `stage/s4_labels/part-{slice}.parquet`。

### S5 · 综合评分 + 分 Tier（`scoring/s5_score.py`，DuckDB 一次性收尾）

硬约束先过滤（文本校验 verdict、无首尾截断、说话人 verdict 合格），再算
`selection_score = 0.35·PQ + 0.25·prosody_dsp + 0.30·expressiveness + 0.10·CE`：
Tier-S = 前 20% 且情感强度 ≥3 且非 neutral；Tier-A/B 按表现力切分。输出扁平 parquet +
每 clip 发布 JSON（§7 schema）。改权重/阈值 = 改 SQL 重跑分钟级。

### HF 发布（`scoring/phase1_hf.py` / `hf_package.py`，一库两视图）

同一个 repo（`leeoxiang/emilia-expressive-zh`）两个 revision：

- **`phase1-filtered` 分支**（短路发布，不等 Phase-2）：WebDataset shard
  （`{clip_id}.flac + {clip_id}.json`，trim 已应用）+ `metadata/phase1_metrics.parquet`
  （每 clip 一行全量 S0-S3 指标，`clip_id` 主键）；
- **Phase-2 标签增量**：`upload_s4_labels()` 只上传 `metadata/s4_labels.parquet`
  （同 `clip_id` 键，随打标进度反复执行），**音频 tar 永不重写**；
- **`main` 分支**：S5 完成后的完整分层版。

## 磁盘布局与幂等

所有产物按 `paths.root` 组织（见 `configs/pipeline_v1.yaml`）：`stage/` 各级 parquet、
`repacked/`、`manifests/`、`done/`（完成标记）、`failed/`、`export/`。写入纪律：先写
`*.tmp` 再 rename，done marker 最后落；重跑 = 全集 − done 集，天然断点续跑。

## 快速开始

```bash
uv venv .venv --python 3.11
uv pip install -p .venv/bin/python -e ".[gpu,audio,data,test]" \
    huggingface_hub pandas audiobox-aesthetics modelscope torchaudio \
    addict simplejson sortedcontainers datasets pillow

.venv/bin/python -m pytest tests/ -q        # 单测（全 mock，~9s）

# 真跑（权重路径与数据根见 configs/pilot.yaml；runbook 有权重下载方法）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m emilia_pipeline.phase1.worker \
    --config configs/pilot.yaml            # 调试: --no-parallel --limit 1

# 磁盘不够装全量源数据时的分批下载-处理-删源循环
.venv/bin/python -m emilia_pipeline.scoring.batch_process \
    --config configs/pilot.yaml --shards ZH-B000000..ZH-B000019 --batch-size 5
```
