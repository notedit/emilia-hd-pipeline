# 本地使用 Phase-1 发布数据（emilia-expressive-zh）

发布数据在本机的位置（与上传到 HF 的目录结构完全一致，本地/远端用法互通）：

```
/opt/dlami/nvme/leolxliu/emilia-zh-full/export/hf_phase1/
├── data/
│   ├── prime/shard-w00-00000.tar ...   # S3 通过 ∧ 韵律 top 40%（精品核心集, ~3,100h）
│   ├── extended/...                    # S3 通过、韵律分在线下（干净但平淡, ~4,000h）
│   └── s3rejected/...                  # S3 声纹拒绝, 原样未剪裁（风险自担, ~5,000h）
├── metadata/phase1_metrics.parquet     # 每 clip 一行全量 S0–S3 指标（打包结束时生成）
├── phase1_manifest_v1.parquet          # clip_id → 所在 shard 的映射（打包结束时生成）
└── README.md                           # dataset card
```

每个 shard 是 ~1GB 的 **WebDataset** tar：`{clip_id}.mp3` + `{clip_id}.json`
（`shard-wNN-*` 的 `wNN` 是并行打包 worker 编号，无语义）。全库统一 mp3：
原样 clip 是 Emilia 源字节，S3 修剪过的 clip（约 12%）以高码率 VBR mp3 重编码。
音频保留**原始采样率**（混采 24/32/44.1 kHz，训练前自行统一重采样）。JSON 含
`text / duration_s / speaker / tier` 及嵌套的 `metrics` 等字段。

> 打包是流式进行的：`data/*/shard-*.tar` 落盘一个可用一个（写入是 tmp+rename
> 原子操作，看到的 tar 一定是完整的）；两个 parquet 在最后才生成。

## 1. 读音频

### 方式 A：HF `datasets`（shard 已全 mp3 同质，webdataset builder 可加载）

```python
from datasets import load_dataset

ROOT = "/opt/dlami/nvme/leolxliu/emilia-zh-full/export/hf_phase1"
ds = load_dataset("webdataset",
                  data_files={"train": f"{ROOT}/data/prime/*.tar"},
                  split="train", streaming=True)
for s in ds:
    audio = s["mp3"]        # 解码为 {array, sampling_rate}
    meta = s["json"]
    break
```

注意：新版 `datasets` 的音频解码走 torchcodec，需要 `pip install torchcodec`
且系统装有 **FFmpeg 动态库**。本机（当前这台 H200 机器）没有 FFmpeg，
`datasets` 解码音频会报 `ImportError: ... install 'torchcodec'`——本机请用方式 B；
方式 A 供有正常 FFmpeg 环境的下游用户使用。

### 方式 B：`webdataset` 库 + soundfile（本机实测可用，训练推荐）

```python
import io, glob, soundfile as sf
import webdataset as wds

ROOT = "/opt/dlami/nvme/leolxliu/emilia-zh-full/export/hf_phase1"

def decode_audio(key, data):
    if key.split(".")[-1].lower() not in ("mp3", "flac", "wav", "ogg"):
        return None
    return sf.read(io.BytesIO(data), dtype="float32")   # -> (ndarray, sr)

ds = (wds.WebDataset(sorted(glob.glob(f"{ROOT}/data/prime/*.tar")),
                     shardshuffle=100)   # shard 级打散
      .shuffle(1000)                     # 样本级打散缓冲
      .decode(decode_audio)
      .map(lambda s: {"id": s["__key__"], "audio": s["mp3"][0],
                      "sr": s["mp3"][1], "meta": s["json"]}))

for s in ds:
    print(s["id"], s["sr"], len(s["audio"]) / s["sr"], s["meta"]["text"][:20])
    break
```

本机不要用 `wds.torch_audio` 解码（torchaudio 同样缺 torchcodec 后端），
用上面的 soundfile 解码器。接 PyTorch 训练用 `wds.WebLoader`：

```python
loader = wds.WebLoader(ds.batched(16, collation_fn=None),
                       batch_size=None, num_workers=4)
```

## 2. 查指标 / 自定义子集：parquet + DuckDB

打包结束后（`metadata/phase1_metrics.parquet` 出现即可用）：

```python
import duckdb

M = f"{ROOT}/metadata/phase1_metrics.parquet"
# 比 prime 更严的自选子集: 只要 clip_id 列表, 音频不动
ids = duckdb.sql(f"""
    SELECT clip_id FROM '{M}'
    WHERE tier = 'prime' AND aes_pq >= 7.5 AND snr_db >= 25
""").df()

# clip_id → 所在 tar, 用 manifest 反查, 只读命中的 shard
# (shard 列是相对 data/ 的路径, 如 "prime/shard-w00-00000.tar")
duckdb.sql(f"""
    SELECT shard, count(*) FROM '{ROOT}/phase1_manifest_v1.parquet'
    WHERE clip_id IN (SELECT clip_id FROM '{M}' WHERE aes_pq >= 7.5)
    GROUP BY 1 ORDER BY 1
""")
```

parquet 用 HF datasets 加载也没问题：

```python
from datasets import load_dataset
meta = load_dataset("parquet", data_files=M, split="train")
```

打包完成前想先查指标，直接用 stage parquet（全量 1997 万行、含被拒行）：

```
/opt/dlami/nvme/leolxliu/emilia-zh-full/stage/{s0_prefilter,s1_acoustics,s2_prosody,s3_speaker/features}/part-*.parquet
```

## 3. 采样率提醒

`sample_rate` 列在 metrics parquet 里；分布约为 32k 为主、混有 24k/44.1k。
喂模型前统一重采样（如 `soxr`/`librosa.resample`），不要假设整库同采样率。
