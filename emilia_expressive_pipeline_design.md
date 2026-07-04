# Emilia 高表现力子集抽取 Pipeline — 落地设计方案

版本: v1.1（简化版：移除全局聚类 S3g；Qwen3-Omni 改用 API） · 状态: 待评审 · 适用: VoxSift / Emilia-ZH

---

## 1. 目标与非目标

**目标**：从 Emilia（中文子集，约 5 万小时量级）中抽取声学干净、韵律动态丰富、情感表达充分的高质量子集，并为每条数据产出情感 / 韵律 / 语境 / 语种等结构化标签，最终以分层（Tier-S / A / B）形式发布到 HuggingFace dataset，供情感 TTS 训练与 prosody 后训练（DPO / GRPO reward 分层）使用。

**非目标**：不做常态化流式清洗服务（一次性批处理项目，未来 VoxSift 常态管线可复用组件）；不做多说话人对话数据的保留与标注（本期直接剔除）；不引入有状态数据库服务（LanceDB / 向量库均不使用）；**不做全局说话人聚类**（不追求跨 clip 归并说话人身份，采样时直接复用 Emilia 自带的 `original_speaker` 分组即可，省去千万级聚类的工程与成本）；**不自建 Qwen3-Omni 推理服务**（S4 打标直接调用云端 API，省去 vLLM 部署、多卡运维与显存调优）。

**关键设计原则**（来自前期讨论的结论，直接作为约束）：

1. 漏斗式级联：便宜的在前，Qwen3-Omni 打标最贵、放最后统一批处理。
2. S4（omni）只做必须"听懂内容"的任务：情感 / 韵律 / 语境 / 文本校验与标点 / 语种。说话人相关问题（串音、overlap）在声纹域（CAM++ 滑窗自一致性）解决，不进 omni schema。本期不做全局说话人聚类（见非目标），说话人身份沿用 Emilia 自带的 `original_speaker` 分组。
3. 两阶段批处理 + anytime 特性：Phase 1 全量扫描出特征，Phase 2 按"最有戏优先"的顺序打标，任意时刻中断，已完成部分即当前最优子集。
4. 存储为纯文件：append-only Parquet 分 stage 落盘，DuckDB 做查询引擎，npy 存 embedding。无锁、无事务、可 rsync。
5. 一切数值保留原值入库，剔除判定只是查询条件；阈值变更 = 改 SQL，不重跑管线。

---

## 2. 总体架构与执行模型

```
Phase 1（全量扫描, GPU×N + CPU 池, 2–4 天）
  S0 元数据预筛 ──► 融合扫描 Worker（每 shard 一个任务）:
                     读 tar → 解码 → [S1 声学 | S2 韵律DSP | S3 滑窗声纹纯度]
                     → part parquet + emb npy（短路: S1 fail 不算 S2/S3）
  ──► 重打包: 存活 clip 按打标优先级写新 WebDataset shard

Phase 2（统一打标, 云端 API, 3–7 天, anytime 可停）
  S4 Qwen3-Omni 结构化打标（worklist 分片认领, asyncio client → 云端 API）

收尾
  S5 综合评分 + 分 Tier（DuckDB SQL, 分钟级）
  ──► HF 打包发布（shuffle → tar+parquet → upload_large_folder）
```

任务粒度与幂等：Phase 1 的任务单位是 **Emilia 原始 tar shard**，Phase 2 的任务单位是 **worklist 分片（默认 5,000 clip/片）**。每个任务的输出是一组以任务 ID 命名的 part 文件，"完成"由 done marker 文件定义。任何 worker 崩溃后重启，pending 集合 = 全集 − done 集合，天然断点续跑；重复执行同一任务是幂等的（同名文件原子覆盖）。

---

## 3. 磁盘数据布局

```
/data/emilia-expressive/
├── configs/pipeline_v1.yaml            # 全部阈值、模型版本、路径（见 §9）
├── source/                             # Emilia 原始 shards（只读挂载）
├── stage/
│   ├── s0_prefilter/part-{shard}.parquet
│   ├── s1_acoustics/part-{shard}.parquet
│   ├── s2_prosody/part-{shard}.parquet
│   ├── s3_speaker/
│   │   ├── features/part-{shard}.parquet      # purity_check 各字段 + emb 指针
│   │   └── embeddings/emb-{shard}.npy          # (n_clips, 192) fp16, 窗均值（备查/未来复用）
│   └── s4_labels/part-{slice}.parquet
├── repacked/                           # Phase 2 输入: 按优先级排序的新 shard
│   ├── shard-{00000}.tar
│   └── repack_index.parquet            # clip_id → (shard, offset)
├── manifests/
│   ├── s4_worklist_v1.parquet          # 排好优先级的待打标清单 + slice_id
│   └── packing_manifest_v1.parquet     # HF 打包清单
├── done/                               # done markers: {stage}/{task_id}
├── failed/                             # 失败记录: {stage}/{task_id}.json
└── export/                             # HF 发布物暂存
```

写入纪律（全局强制）：所有 parquet / npy 先写 `*.tmp` 再 `os.rename`；done marker 在数据文件 rename 成功后才创建。查询侧只认非 tmp 文件。

Embedding 不进 parquet：clip 级 embedding 存 `emb-{shard}.npy`，parquet 里记 `emb_file + emb_row`。滑窗级 embedding 仅 S3 纯度判决时在 worker 内使用，判决结论落 parquet 后即丢弃，不持久化。clip 级均值 embedding 保留备查（未来若需做说话人复用可直接读取）。

---

## 4. Stage 规格

### S0 · 元数据预筛（纯 metadata，不解码音频）

输入 Emilia 自带 JSON 字段，输出每 shard 的 clip 白名单（含拒绝原因）。规则：

| 字段 | 条件 | 说明 |
|---|---|---|
| duration | 3.0 ≤ d ≤ 20.0 s | 短于 3s 无韵律结构；长于 20s 情感不纯 |
| language | == "zh" | 本期只做中文 |
| dnsmos (原始) | ≥ 3.2 | 比 Emilia 原始 3.0 收紧的粗刀 |
| text | 非空, 长度 ≥ 4 字 | 空文本无法做校验 |

实现为 Phase 1 融合 worker 的第一步（内联），不单独跑一遍。

### S1 · 严格声学过滤

GPU 模型两个 + CPU 指标若干，全部数值入库，pass 判定为查询条件：

| 指标 | 模型/方法 | 严格档阈值（初值, 待 §10 校准） |
|---|---|---|
| aesthetics PQ | Audiobox-Aesthetics | ≥ 7.0（主闸门, 对源分离伪影敏感） |
| aesthetics PC | 同上 | ≤ 2.5（低复杂度 = 干净单人语音） |
| aesthetics CE/CU | 同上 | 不设硬阈值, 入库供 S5 评分 |
| DNSMOS P.835 OVRL | DNSMOS onnx | ≥ 3.5 |
| SNR 估计 | WADA / 能量法, CPU | ≥ 20 dB |
| clipping_ratio | CPU | ≤ 0.001 |
| 有效带宽 | 频谱 roll-off, CPU | ≥ 8 kHz（防上采样假 24k） |

短路规则：S1 判 fail 的 clip 不再计算 S2 / S3（省 CPU 与 CAM++ 前向），但 S1 自身所有指标必须算全（供 ROC 校准与阈值回放）。

### S2 · 韵律丰富度粗筛（CPU DSP）

目的：判"有没有戏"，砍掉平淡朗读，故意宽松（精细判断留给 S4）。F0 用 pyworld（CPU 便宜、对 24k 直接可用），VAD 复用 silero。

| 指标 | 定义 | 备注 |
|---|---|---|
| f0_std_st / f0_range_st | log-F0 标准差 / P5–P95 范围（半音） | 跨性别可比; 平淡朗读常 <2st |
| energy_std_db | 帧能量标准差 | |
| speech_rate_cps / rate_var_cv | 字/秒 及 滑窗语速变异系数 | 字数来自 Emilia text |
| pause_count / pause_total_ms | VAD 停顿统计 | |
| f0_tracker_confidence | voiced 帧占比 + 轨迹跳变率 | **透传给 S3 overlap 判决** |
| prosody_dsp_score | 上述指标 z-score 加权和 (权重进 config) | 取全体存活样本 top 40% |

top 40% 的分位阈值在 Phase 1 全量跑完后由 DuckDB 一次性计算（这是"数值入库、判定后置"原则的直接受益：无需预估分布）。

### S3 · 声纹：滑窗纯度检测

**S3a 滑窗特征（融合 worker 内, GPU）**：CAM++ 以 1.5s 窗、50% overlap 提取窗序列 embedding；clip 级 embedding = 窗均值。产出自一致性指标：

- `mean_win_cos` / `min_win_cos`：各窗与窗均值中心的 cosine 均值 / 最小值；
- 低于阈值的连续窗段定位 → `intrusion_span_ms`。

本期不做全局聚类：说话人身份直接沿用 Emilia 自带的 `original_speaker` 字段，S3 只负责判"这条 clip 内部是否纯净（单说话人、无串音/overlap）"，不追求跨 clip 归并。clip 级均值 embedding 落 npy 备查，供未来需要时复用。

**S3b 纯度判决**（融合 worker 内在线判，纯窗序列运算，无需聚类）：判据完全基于 clip 内部的窗 cosine 形态与 F0 轨迹，不依赖任何全局质心。

| 窗 cosine 形态 | f0_tracker_confidence | verdict |
|---|---|---|
| 局部连续塌陷 | — | intrusion → 首尾段修剪回收(剩余≥3s) / 中段 rejected |
| 均匀压低（窗间离散大） | 差 | overlap_rejected |
| 均匀压低 | 正常 | degraded_pass（放行, 交给 S1/S5 分数体系） |
| 正常 | 正常 | single |

verdict 词表: `single / intruded_trimmed / intruded_rejected / overlap_rejected / degraded_pass`。被修剪的 clip 更新时长并记 `s3_trim`；轻度 overlap 的最终防线是 S1 的 PC≤2.5 与 Tier-S 人工抽检，本级阈值向"不误杀"偏。

### 重打包（Phase 1 → Phase 2 的桥）

从 DuckDB 查询存活集，按打标优先级 `priority = prosody_dsp_score × norm(aesthetics_pq)` 降序写新 WebDataset shard（约 1GB/个），同时生成 `repack_index.parquet` 与 `s4_worklist_v1.parquet`（含 slice_id 分片，默认 5,000 clip/片，按优先级顺序编号——**slice 编号顺序即打标顺序，anytime 特性由此保证**）。多进程并行：按 slice 分派，每进程独立产出自己的 tar；读取原始 tar 时清单按源 shard 分组排序，保持顺序读。

---

## 5. S4 · Qwen3-Omni 打标详设

### 5.1 服务（云端 API，不自建部署）

- 模型：`Qwen3-Omni-30B-A3B-Instruct`（阿里云百炼 / DashScope 提供的多模态 API；若届时 Qwen3.5-Omni API 上线且中文横评不差，改个 model 名即可切换——判官接口模型无关）。
- 调用方式：直接调云端 OpenAI-兼容接口（DashScope `compatible-mode` endpoint），无需 vLLM、无需本地 GPU、无显存与并发调优。API key 与 endpoint 进 config。
- 关键参数：greedy（`temperature=0`）、`response_format` 走 JSON（若 API 支持 guided/structured output 则用 schema 约束，见 5.3；否则用 prompt 强约束 + 客户端 schema 校验兜底）。
- 音频输入：client 侧将 24k flac 解码重采样为 API 要求的采样率后，以 base64 data-URI 或可访问 URL 传入（按 DashScope 音频输入规范）。
- 成本与限流：按 token 计费，成本随打标量线性增长——这正是漏斗式级联把 omni 放最后、且 S0–S3 尽量收紧的原因。并发受 API 侧 QPS/RPM 限额约束（见 6.3），不再是显存瓶颈。

### 5.2 一遍 vs 两遍

默认**单遍全量打标**。开跑前用 5,000 条 pilot 实测 triage 通过率：若 <60%，切换两遍制（triage 输出 ~30 token，通过者进全量打标，省下大量输出 token 与费用）；若 ≥60%（预期如此，因 S2 已按韵律预筛过），两遍的重复输入 token（音频重复上传）抵消其收益，维持单遍。该开关在 config 中，两套 prompt 均预先准备。

### 5.3 输出 Schema（guided JSON, 全封闭词表）

```json
{
  "text_verdict": "match | fixable | broken",
  "text_fixed": "string",
  "text_punctuated": "string",
  "emotion": {
    "primary": "neutral|happy|excited|sad|angry|fearful|surprised|disgusted|affectionate|serious",
    "secondary": "同词表或 null",
    "intensity": 1-5,
    "confidence": 0.0-1.0
  },
  "prosody": {
    "expressiveness": 1-5,
    "speaking_style": "narration|conversational|storytelling|speech|broadcast|acting|vlog|interview",
    "rhythm": "steady|varied|dramatic",
    "prominent_stress": true|false
  },
  "context": {
    "scenario": "podcast|audiobook|drama|interview|lecture|vlog|customer_service|other",
    "register": "formal|casual|intimate",
    "summary": "一句话内容摘要"
  },
  "language": {
    "primary": "zh|...",
    "code_switch": true|false,
    "accent": "standard|accented|dialect"
  },
  "paralinguistic": ["laughter|sigh|crying|breath_prominent|filler_heavy|disfluent"],
  "defects": ["truncated_head|truncated_tail|artifact|other"],
  "usable": true|false
}
```

注意：schema 不含任何说话人相关字段（原则 2）；不含数值音质分（数值分归专用打分器）。

### 5.4 Prompt 要点

System：角色设定为"中文语音数据标注专家"，逐字段给出判据定义（尤其 intensity 与 expressiveness 的 1–5 锚点描述）。User：参考文本 + 音频 + 指令。Few-shot 5–8 条覆盖易混 case：激动 vs 愤怒、演绎 vs 朗读、code-switch、疑问 vs 反问的标点。**量表锚点样例**（"这是 expressiveness=2 的例子 / 这是 5 的例子"）用于对齐分布，是 §10 校准后的主要调整手段。prompt 全文与 few-shot 音频 hash 进 `prompt_version`，不同版本标签不可混用。

---

## 6. 并行与调度框架（多进程 · 多卡 · 可多机）

### 6.1 任务认领：文件系统为准 + Redis 派活

真相源永远是文件系统（done markers），Redis 只是派活的信箱，丢了可重建：

```python
# dispatch.py —— 幂等派活
all_tasks   = enumerate_tasks(stage)                  # shard 列表 或 worklist slice 列表
done        = {p.stem for p in (DONE_DIR/stage).iterdir()}
pending     = sorted(all_tasks - done)                # Phase2: 按 slice 编号 = 优先级顺序
r = redis.Redis(...)
r.delete(f"q:{stage}")
r.rpush(f"q:{stage}", *pending)
```

Worker 主循环（Phase 1 / Phase 2 通用骨架）：

```python
while (task_id := r.lpop(f"q:{stage}")):
    if (DONE_DIR/stage/task_id).exists():          # 双保险
        continue
    try:
        process(task_id)                            # 内部原子写 part 文件
        (DONE_DIR/stage/task_id).touch()
    except Exception as e:
        write_json(FAILED_DIR/stage/f"{task_id}.json", {"err": str(e), "ts": now()})
        # 不重回队列; 失败集由 dispatch 重跑时自然重新入队
```

单机场景 Redis 可换成 `multiprocessing.Queue`，接口一致；多机直接共享 Redis 与 NFS/JuiceFS 挂载的数据目录（你已有 JuiceFS 评估结论，可直接落）。无心跳、无租约——任务幂等使得"偶尔重复执行"无害，比租约机制简单一个量级。

### 6.2 Phase 1 融合 Worker 的进程结构（每 GPU 一个）

```
主进程 (CUDA_VISIBLE_DEVICES=k)
├── 常驻 GPU 模型: Audiobox-Aesthetics, DNSMOS(onnx-gpu), CAM++
├── CPU 侧: mp.Pool(n_cpu_per_gpu, spawn)  ← 解码/重采样/VAD/DSP(pyworld)/SNR
└── 流水线:
     tar 顺序读 → S0 白名单过滤
       → CPU 池 imap: decode + s2_dsp + s1_cpu_metrics   (乱序返回, 攒 batch)
       → GPU: aesthetics + dnsmos 批推理 → S1 判定(短路)
       → GPU: CAM++ 滑窗批推理 (仅 S1 pass)
       → 三个 stage 的行缓冲 → shard 结束时一次性原子写 3 个 parquet + 1 个 npy
```

要点：CPU 池大小 = `total_cores / n_gpus`，用 spawn 避免 CUDA fork 问题；GPU batch 按解码后音频攒（Aesthetics/DNSMOS 定长切窗，CAM++ 按窗展平后统一 batch）；单 shard 内全部数据在内存中完成、shard 级落盘，无跨任务共享状态。启动脚本：

```bash
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python phase1_worker.py --config pipeline_v1.yaml &
done
```

### 6.3 Phase 2 S4 Client（asyncio, 直连云端 API）

```
云端 Qwen3-Omni API (OpenAI-兼容 endpoint)
S4 client (每 slice 一个任务, 可多进程并行认领):
  按 repack_index 顺序读音频 → 重采样/base64 → 构造 messages
  → asyncio + aiohttp, 全局 semaphore(=API 限额内的最大在飞并发)
  → 响应 JSON 校验(schema) → 行缓冲 → slice 结束原子写 part-{slice}.parquet
失败处理: 单条重试(指数退避, 覆盖 429/5xx 限流); 仍失败写入行内 error 字段并标 s4_status=failed(不丢弃)
```

Client 是纯 CPU/IO 进程，无本地 GPU 依赖。唯一需调优的参数是**在飞并发数（semaphore 上限）**，取 API 账号的 RPM/TPM 限额与观测到的 429 频率决定；触发限流即退避降速。**打标进度即 anytime 检查点**：任意时刻 `done/s4/` 里的 slice 数 × 5,000 = 已完成的最优子集规模。

### 6.4 监控（够用即可, 不引平台）

一个 cron 脚本每 30 分钟产出快照：各 stage done/failed/pending 计数、S4 已消耗 token 数与累计费用估算、429 限流率、滚动窗口的各 verdict/reject 分布。分布突变（例如某小时 emotion=neutral 占比骤升）是 prompt 回归或数据源异常的信号，人工介入。

---

## 7. 单条数据最终 Meta Schema

发布态（tar 内 `{clip_id}.json`）与内部 parquet 拍平（`omni_labels.emotion.primary → emotion_primary`）共用同一定义：

```json
{
  "clip_id": "emilia_zh_B0000412_S03_0087",
  "schema_version": "1.3",
  "audio": {"path": "...", "duration_s": 8.42, "sample_rate": 24000,
            "channels": 1, "loudness_lufs": -19.3},
  "source": {"dataset": "emilia", "dataset_version": "emilia-large-v1",
             "original_id": "ZH_B0000412_S03_W000087",
             "original_text": "...", "original_speaker": "ZH_B0000412_S03",
             "original_dnsmos": 3.41, "original_language": "zh"},
  "acoustics": {"aesthetics": {"pq": 7.8, "pc": 1.9, "ce": 7.1, "cu": 7.4},
                "dnsmos_p835": {"sig": 3.9, "bak": 4.1, "ovrl": 3.7},
                "snr_db": 28.4, "clipping_ratio": 0.0, "bandwidth_hz": 11200},
  "prosody_dsp": {"f0_mean_hz": 218.5, "f0_std_st": 4.2, "f0_range_st": 14.1,
                  "energy_std_db": 6.8, "speech_rate_cps": 4.9, "rate_var_cv": 0.31,
                  "pause_count": 2, "pause_total_ms": 610,
                  "f0_tracker_confidence": 0.94, "prosody_dsp_score": 0.81},
  "speaker": {"original_speaker": "ZH_B0000412_S03",
              "embedding_ref": {"emb_file": "emb-001234.npy", "emb_row": 187},
              "gender_pred": "female",
              "purity_check": {"n_windows": 9, "mean_win_cos": 0.91, "min_win_cos": 0.83,
                               "f0_stability": 0.94,
                               "verdict": "single", "intrusion_span_ms": null,
                               "trimmed": false}},
  "omni_labels": {"model": "qwen3-omni-30b-a3b-instruct", "prompt_version": "v3.1",
                  "text_verdict": "fixable",
                  "text_fixed": "你知道吗，那天我真的差点就哭出来了。",
                  "text_punctuated": "你知道吗？那天我真的……差点就哭出来了。",
                  "cer_vs_original": 0.0,
                  "emotion": {"primary": "sad", "secondary": "affectionate",
                              "intensity": 4, "confidence": 0.86},
                  "prosody": {"expressiveness": 4, "speaking_style": "storytelling",
                              "rhythm": "dramatic", "prominent_stress": true},
                  "context": {"scenario": "podcast", "register": "intimate",
                              "summary": "讲述者回忆一段令自己动情的往事"},
                  "language": {"primary": "zh", "code_switch": false, "accent": "standard"},
                  "paralinguistic": ["breath_prominent"], "defects": [], "usable": true},
  "selection": {"selection_score": 0.87, "tier": "S", "reject_reason": null},
  "pipeline": {"version": "voxsift-emilia-v1.3",
               "stages_passed": ["s0","s1","s2","s3","s4"],
               "processed_at": "2026-07-05T03:12:44Z"}
}
```

### S5 评分与分层（DuckDB, 收尾一次性）

```
selection_score = 0.35·norm(pq) + 0.25·norm(prosody_dsp_score)
                + 0.30·norm(expressiveness × intensity) + 0.10·norm(ce)
硬约束: text_verdict != 'broken' AND 'truncated%' NOT IN defects
        AND verdict IN ('single','intruded_trimmed')
Tier-S: score top 且 emotion != neutral 且 intensity >= 3
Tier-A: 质量达标全情感分布   Tier-B: 达标但表现力一般
采样: 按 original_speaker × emotion 双维分层, 抑制头部说话人垄断
```
