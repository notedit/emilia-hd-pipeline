# Pilot 真跑 Runbook —— 从 1 个真实 Emilia shard 到出 Tier

> 目标：在投入 5 万小时全量之前，用**少量真实数据 + 真实模型/真实 API** 把整条链路跑通一次，
> 校准阈值、验证标签质量、实测成本与吞吐。这是"设计假设"与"现实"的第一次对账。
>
> 适用版本：v1.1（无全局聚类；Qwen3-Omni 走云端 API）。所有命令从项目根
> `/workspace/user_code/emilia-hd-pipeline` 执行，Python 为
> `/data/miniforge/envs/env-system/bin/python`（下文简写 `python`）。

---

## 0. 前置条件检查（P0 先决）

pilot 需要 mock 层之外的三样真实资源。缺哪样，对应阶段就只能跑 mock：

| 资源 | 用途 | 检查命令 | 缺失后果 |
|---|---|---|---|
| **GPU + CUDA 驱动** | S1 Audiobox/DNSMOS、S3 CAM++ 真前向 | `nvidia-smi`；`python -c "import torch;print(torch.cuda.is_available())"` | 本机 driver 12040 太旧，torch 报 False → 只能 mock。需升级驱动或换 torch 版本 |
| **模型权重** | 同上 | 见 §1，路径填进 `configs/pipeline_v1.yaml` 的 `models.*` | `get_model` 自动回退 mock（`_try_build_real_model` 找不到权重就返回 None） |
| **Venus API key** | S4 omni 打标 | `echo $OPENAI_API_KEY` | S4 走 mock transport，标签是 hash 伪造的，不能用于质量评估 |

```bash
# 一次性体检
python - <<'PY'
import torch, os
from emilia_pipeline.common.config import load_config
cfg = load_config("configs/pipeline_v1.yaml")
print("cuda:", torch.cuda.is_available())
print("aesthetics_weights:", cfg.models.aesthetics_weights)
print("dnsmos_onnx:", cfg.models.dnsmos_onnx)
print("campplus_weights:", cfg.models.campplus_weights)
print("api_key set:", bool(cfg.api_key()))
PY
```

> **关键提醒**：只要上面任一项是 mock，产出的对应指标就**不能用于阈值校准或质量结论**。
> pilot 的价值恰恰在真实组件上。先把 P0 资源凑齐，再往下走。

---

## 1. 准备权重与配置

把真实权重放到本地，路径写进 `configs/pipeline_v1.yaml`：

```yaml
models:
  aesthetics_model: audiobox-aesthetics
  aesthetics_weights: /data/weights/audiobox-aesthetics/     # 真实路径
  dnsmos_onnx:        /data/weights/dnsmos/sig_bak_ovr.onnx
  campplus_model:     cam++
  campplus_weights:   /data/weights/campplus/campplus.onnx
runtime:
  use_mocks: false        # 关键：pilot 必须关掉全局 mock 开关
  n_gpus: 2
  audio_sample_rate: 24000
```

> `use_mocks: false` 时，`get_model` 会尝试构建真实模型；**若权重路径不存在会静默回退 mock**。
> 跑之前用上面的体检脚本确认三个权重路径都真实存在，否则你以为在测真模型、其实还是 mock。

准备 pilot 数据目录（1–5 个真实 Emilia tar shard）：

```bash
mkdir -p /data/emilia-pilot/source
cp /path/to/emilia/ZH_B00004.tar /data/emilia-pilot/source/00000.tar   # 1 个起步
# 把 configs 里 paths.* 指向 /data/emilia-pilot/...（或用一份 pilot 专用 yaml）
```

建议复制一份 `configs/pilot.yaml`，把 `paths.root` 及各子路径改到 `/data/emilia-pilot/`，
避免污染正式目录。下文命令用 `CFG=configs/pilot.yaml`。

---

## 2. Phase 1 —— 全量扫描（S0→S1→S2→S3）

Phase-1 worker 是唯一有 CLI 的组件。每卡一个进程；pilot 单卡即可：

```bash
export CFG=configs/pilot.yaml
CUDA_VISIBLE_DEVICES=0 python -m emilia_pipeline.phase1.worker --config $CFG
# 多卡：for g in 0 1; do CUDA_VISIBLE_DEVICES=$g python -m emilia_pipeline.phase1.worker --config $CFG & done; wait
# 调试单 shard、串行 CPU：加 --no-parallel --limit 1
```

产出：`stage/s0..s3/part-00000.parquet` + `stage/s3_speaker/embeddings/emb-00000.npy` +
`done/phase1/00000` marker。

### 要观察的指标（用 DuckDB 直接查 parquet）

```bash
python - <<'PY'
import duckdb, os
from emilia_pipeline.common.config import load_config
from emilia_pipeline.common.io_utils import parquet_glob
cfg = load_config(os.environ["CFG"])
con = duckdb.connect()
g = lambda p: parquet_glob(p)
s0 = g(cfg.paths.s0_prefilter); s1 = g(cfg.paths.s1_acoustics); s3 = g(cfg.paths.s3_speaker_features)
print("=== S0 存活率 ===")
print(con.execute(f"SELECT passed, count(*) FROM '{s0}' GROUP BY passed").fetchall())
print("=== S0 拒绝原因分布 ===")
print(con.execute(f"SELECT reject_reason, count(*) FROM '{s0}' WHERE NOT passed GROUP BY 1 ORDER BY 2 DESC").fetchall())
print("=== S1 存活率 + 各 aesthetics/dnsmos 分位（校准阈值的关键）===")
print(con.execute(f"""SELECT count(*) n,
    avg(passed::int) pass_rate,
    quantile_cont(aes_pq,[0.1,0.5,0.9]) pq_q,
    quantile_cont(aes_pc,[0.1,0.5,0.9]) pc_q,
    quantile_cont(dnsmos_ovrl,[0.1,0.5,0.9]) ovrl_q,
    quantile_cont(snr_db,[0.1,0.5,0.9]) snr_q FROM '{s1}'""").fetchall())
print("=== S3 verdict 分布（串音/裁剪是否合理）===")
print(con.execute(f"SELECT verdict, count(*) FROM '{s3}' GROUP BY 1 ORDER BY 2 DESC").fetchall())
PY
```

**判读**：
- **S0/S1 存活率**：过低（<10%）= 阈值太严，会浪费 omni 预算前就砍光；过高（>80%）= 太松，脏数据进 S4。
- **aes_pq / dnsmos_ovrl 分位**：把 §4 的初值（pq≥7.0, ovrl≥3.5）叠到分布上看落在哪个百分位。
  这是 **P1 阈值校准**的原始输入——若 7.0 卡掉 95%，说明该 shard 整体偏低或阈值需下调。
- **S3 verdict**：`intruded_trimmed` / `overlap_rejected` 占比。真 CAM++ 下 `mean_win_cos`
  能否区分纯净与串音，直接决定 S3 是否有效（mock 下这条形同虚设，pilot 才第一次真验）。

---

## 2b. 小磁盘分批处理（500G 装不下 1.26TB 全量）

Emilia-ZH 正式版 ≈ **920 tar / 1.26TB / ~54,000 小时**，500G 磁盘装不下。但 **Phase-1 产物极小**——
实测每 tar 的 stage parquet+npy 仅 **~8.5MB**，全量才 **~10GB**。所以策略是：

> **源 tar 是"临时的"，stage 产物是"要留的"。** 下一批 → 处理 → 删源 tar + 清 HF 缓存 → 只留 parquet。

`batch_process.py` 把这个循环编排好了（幂等：done marker 已完成的 shard 跳过；磁盘水位守卫拒绝超预算批次）：

```bash
export HF_TOKEN=hf_...  HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60   # Xet 后端易超时，走经典 LFS
python -m emilia_pipeline.scoring.batch_process \
  --config /data/emilia-100h/pilot_100h.yaml \
  --repo amphion/Emilia-Dataset --prefix Emilia/ZH/ \
  --shards ZH-B000000..ZH-B000019 \
  --batch-size 5 --min-free-gb 60 --gpus 0
```

**磁盘账（实测标定）**：

| 项 | 占用 |
|---|---|
| 固定保留（全程）：stage parquet + emb 全量 | ~10 GB |
| 滚动峰值：一批源 tar（+ 同量 HF 缓存，处理后清） | `batch_size × 2GB × 2` |
| 例：`--batch-size 5` | 峰值 ~20GB，远低于 500G |

**关键陷阱（脚本已处理）**：
1. `hf_hub_download` 先下到 `~/.cache/huggingface` 再 copy → 同一 tar 占 2 份。脚本每批 `purge_hf_cache_for` 清掉。
2. Xet 后端（`cas-bridge.xethub.hf.co`）经镜像易 ReadTimeout → 设 `HF_HUB_DISABLE_XET=1` 走经典 LFS + 拉长 timeout。
3. 中断安全：任何时刻挂掉，done marker 保住已完成 shard，重跑自动续。

**全量 920 tar 的量级**：固定产物仍只 ~10GB；处理时间受 DNSMOS(CPU)+CAM++ 限制，实测 ~2-3h/tar-worker × 并行度。可分多次跑（脚本天然续跑），或挂后台。

处理完（或每处理够一批），用 §2 的 DuckDB 查询 / §附录的 `calibrate.py` 看真实分布校准阈值——**源 tar 已删不影响，因为数值全在 parquet 里**（"数值入库、判定后置"原则的直接受益）。

---

## 3. 重打包 —— 存活集 → 优先级 worklist + 新 shard

```bash
python - <<'PY'
import os
from emilia_pipeline.common.config import load_config
from emilia_pipeline.phase1.repack import run_repack
cfg = load_config(os.environ["CFG"])
# pilot 数据少，先关掉 top-40% 分位门，保留全部存活样本便于观察
summary = run_repack(cfg, apply_s2_top_fraction=False, parallel=False)
print(summary)   # {n_survivors, n_slices, n_indexed, worklist_path, index_path}
PY
```

产出：`repacked/shard-00000.tar`（**已应用 S3 裁剪**——修复 #1 后 `intruded_trimmed` 的音频
是裁剪后的）、`repacked/repack_index.parquet`、`manifests/s4_worklist_v1.parquet`。

**要观察**：
- `n_survivors` 是否符合 §2 存活率预期；
- worklist 的 `prosody_dsp_score` 现在是**全局 z-score**（修复 #2），验证它对 shard 无关：
  ```sql
  SELECT priority_rank, clip_id, prosody_dsp_score, priority FROM 'manifests/s4_worklist_v1.parquet'
  ORDER BY priority_rank LIMIT 20;
  ```
- 抽查一个 `trimmed=true` 的 clip，解码 `repacked` 里的音频确认时长 == `trim_end_s - trim_start_s`。

---

## 3b.（可选）Phase-1 过滤子集直发 HuggingFace —— 跳过 Phase 2

如果只想把 **Phase-1 过滤结果**（S0-S3 存活集）直接发到 HF、不等 Phase 2 标注 / S5 打分，
用这条短路路径。它复用 repack 的存活集定义与全局 `prosody_dsp_score`，音频**直接从
`source/*.tar` 读并内联应用 S3 裁剪**（不依赖 `run_repack`），每条 clip 产出
`{clip_id}.flac` + `{clip_id}.json`（含完整 S0-S3 指标块）。

```bash
python - <<'PY'
import os
from emilia_pipeline.common.config import load_config
from emilia_pipeline.scoring import phase1_hf
cfg = load_config(os.environ["CFG"])
# pilot：关掉 top-40% 分位门看全量；小 shard 便于观察
res = phase1_hf.package_phase1(cfg, apply_s2_top_fraction=False, target_shard_bytes=100_000_000)
print(res.n_clips, res.n_with_audio, res.export_dir)
# 真上传需 HF_TOKEN + 配置 hf.repo_id；缺任一则只产出 export/hf_phase1/ 不上传。
# 与最终标注版共用同一 repo，发到 hf.phase1_revision 分支（默认 phase1-filtered）。
# phase1_hf.upload_phase1_to_hf(cfg)   # 或传 repo_id=/revision= 覆盖
PY
```

产出：`export/hf_phase1/data/shard-*.tar`、`export/hf_phase1/phase1_manifest_v1.parquet`、
`export/hf_phase1/README.md`（自动生成的 dataset card，含过滤漏斗与 schema 说明）。
配置见 `configs/pipeline_v1.yaml` 的 `hf:` 段（`repo_id` / `private` / `phase1_revision` /
`shard_bytes` / `include_metrics` / `apply_s2_top_fraction`）。

**要观察**：shard 内每条 clip 是否都有 `.flac` + `.json` 两个 member；`README.md` 里的
clip 数与漏斗阈值是否符合预期;`upload_phase1_to_hf` 在缺 `HF_TOKEN` / `hf.repo_id` 时应优雅跳过。


---

## 4. Phase 2 —— S4 Qwen3-Omni 真调用（最贵的一步）

S4 走公司内部 **Venus LLM 代理**（OpenAI 兼容端点，专有 `venus_multimodal_url` 音频类型），
默认 config 已指向它：`model: server:272349`、`base_url: http://v2.open.venus.oa.com/llmproxy`、
key 读环境变量 `OPENAI_API_KEY`（base_url 可用 `OPENAI_BASE_URL` 覆盖）。

```bash
export OPENAI_API_KEY=<你的 Venus key>
# 可选：export OPENAI_BASE_URL=http://v2.open.venus.oa.com/llmproxy
python - <<'PY'
import os
from emilia_pipeline.common.config import load_config
from emilia_pipeline.phase2.s4_client import run_s4_phase
cfg = load_config(os.environ["CFG"])
assert cfg.api_key(), "OPENAI_API_KEY 未设置——会走 mock，标签不可用！"
# run_s4_phase 会：先跑 slice 0 做 pilot 单遍，测 pass-rate，
# 据此决定其余 slice 单遍/两遍（修复 #7：pilot 自适应已接线）
summary = run_s4_phase(cfg)
print(summary)   # {n_slices, pilot_pass_rate, two_pass, completed}
PY
```

> 若在交互式 session 里缺登录/凭证，可用 `! <command>` 前缀直接在本会话跑（如 `! echo $OPENAI_API_KEY`）。
> 切换到其它 OpenAI 兼容端点（如 DashScope）：把 `configs` 里 `s4.provider` 改为 `openai`
> （改用标准 `input_audio` 音频类型），并相应改 `model` / `base_url` / `api_key_env`。

产出：`stage/s4_labels/part-{slice}.parquet` + `done/s4/{slice}` markers。

**要观察 / 记录（P1 成本吞吐 + 质量）**：
- **pilot_pass_rate**：<60% 会自动切两遍制（§5.2）。记录实际值。
- **单条延迟 & 并发**：调 `s4.max_concurrency`，观察是否触发 429（退避已实现）。
  从小并发起步（如 4），逐步加到不被限流的上限。
- **成本**：跑完用 monitor（§6）看 token 消耗与费用估算。**外推到全量**：
  `全量成本 ≈ (pilot 成本 / pilot clip 数) × 预估全量存活 clip 数`。
- **标签质量（人工）**：抽 20–50 条**真听音频**核对 omni 标签——尤其易混对
  （激动 vs 愤怒、演绎 vs 朗读、疑问 vs 反问标点）。这是**金标集**的雏形。

```bash
# 导出若干条标签供人工审阅
python - <<'PY'
import os, duckdb
from emilia_pipeline.common.config import load_config
from emilia_pipeline.common.io_utils import parquet_glob
cfg = load_config(os.environ["CFG"])
con = duckdb.connect()
g = parquet_glob(cfg.paths.s4_labels)
print(con.execute(f"""SELECT clip_id, s4_status,
    labels.emotion.primary, labels.emotion.intensity,
    labels.prosody.expressiveness, labels.text_verdict, labels.text_punctuated
    FROM '{g}' WHERE s4_status='ok' LIMIT 30""").fetchall())
PY
```

---

## 5. S5 —— 评分 + 分层 + 出 Tier

```bash
python - <<'PY'
import os
from emilia_pipeline.common.config import load_config
from emilia_pipeline.scoring.s5_score import run_s5
cfg = load_config(os.environ["CFG"])
res = run_s5(cfg, write=True)
print("candidates:", res.n_candidates, "kept:", res.n_kept)
print("tier_counts:", res.tier_counts)     # {'S':.., 'A':.., 'B':..}
PY
```

产出：`export/meta/part-all.parquet`（全量，含被拒）+ `export/meta_json/{clip_id}.json`（仅入选）。

**要观察**：
- **Tier 分布**：S/A/B 比例。注意修复 #3 后 **A/B 按 expressiveness 分**
  （B=表现力一般 `<=tier_b_max_expressiveness`，A=其余），neutral 只是 Tier-S 的排除门。
- **人工听 Tier-S 抽检**：随机放 10 条 Tier-S，确认"确实高表现力、情感饱满"。
  这是分层合理性的最终验收。
- **published JSON 的 `audio.loudness_lufs`**：修复 #4 后应是真实值（非 0.0）；
  `duration_s` 对裁剪过的 clip 应等于裁剪后时长且与音频一致。

---

## 6. 监控快照 —— 成本 / 分布 / 429

```bash
python - <<'PY'
import os
from emilia_pipeline.common.config import load_config
from emilia_pipeline.scoring import monitor
cfg = load_config(os.environ["CFG"])
snap = monitor.build_snapshot(cfg)
print(monitor.format_snapshot(snap))
# 生产环境挂 cron 每 30 分钟：monitor.write_snapshot(cfg)
PY
```

关注：各 stage done/failed/pending 计数、S4 token 与费用估算、429 率、
verdict/emotion/tier 分布。**分布突变**（如某段 emotion=neutral 骤升）是 prompt 回归或数据源异常的信号。

---

## 7.（可选）HF 打包

```bash
python - <<'PY'
import os
from emilia_pipeline.common.config import load_config
from emilia_pipeline.scoring import hf_package
cfg = load_config(os.environ["CFG"])
res = hf_package.package_export(cfg, target_shard_bytes=100_000_000)  # pilot 用小 shard
print("n_clips:", res.n_clips, "shards:", len(res.shard_paths))
# 真上传需 HF_TOKEN；缺失则只产出 export/ 不上传
# hf_package.upload_to_hf(cfg, "org/emilia-expressive-zh")
PY
```

---

## 8. Pilot 验收清单（跑完逐项确认）

- [ ] Phase-1 三个真实模型（Audiobox/DNSMOS/CAM++）真的跑了（非 mock），指标有限且分布合理
- [ ] S0/S1 存活率在合理区间；记录 aes_pq/dnsmos_ovrl/snr 分位 → 输入 P1 阈值校准
- [ ] S3 verdict 分布合理；抽查 `intruded_trimmed` 音频确认已裁剪（时长匹配）
- [ ] worklist `prosody_dsp_score` 为全局 z-score（对 shard 无关）
- [ ] S4 真调 Venus 代理成功；记录 pilot_pass_rate、单条延迟、并发上限、429 率
- [ ] 人工审阅 20–50 条 omni 标签质量（易混对）→ 金标集雏形
- [ ] S5 Tier 分布合理；人工听 Tier-S 抽检通过；loudness/duration 正确
- [ ] 记录**单位成本**（元/千条）与**吞吐**（clip/GPU 小时）→ 外推全量，校验 §2 时间/预算假设

---

## 9. 已知工程缺口（pilot 会暴露，正式化前补）

- **CLI 覆盖**：目前只有 Phase-1 worker 有 `__main__`；repack/S4/S5/monitor 靠 `python -c` 驱动。
  正式化建议加薄 CLI（`python -m emilia_pipeline.phase2.s4 --config ...` 等）。
- **CUDA 驱动**：本机 driver 12040 对当前 torch 太旧 → GPU 路径需先解决（升级驱动或换 torch）。
- **阈值仍为初值**：§4 所有阈值待用 pilot 分布 + 金标集做 ROC 校准（这是 pilot 后的首要工作）。
- **多机/JuiceFS、崩溃恢复、长稳**：pilot 单机跑通后再做规模化与鲁棒性实测。
