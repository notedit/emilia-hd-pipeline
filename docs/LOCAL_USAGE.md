# 本地使用 Phase-1 发布数据（emilia-expressive-zh）

发布数据在本机的位置（与上传到 HF 的目录结构完全一致，本地/远端用法互通）：

```
/opt/dlami/nvme/leolxliu/emilia-zh-full/export/hf_phase1/
├── data/
│   ├── prime/shard-00000.tar ...       # S3 通过 ∧ 韵律 top 40%（精品核心集, ~3,100h）
│   ├── extended/shard-00000.tar ...    # S3 通过、韵律分在线下（干净但平淡, ~4,000h）
│   └── s3rejected/shard-00000.tar ...  # S3 声纹拒绝, 原样未剪裁（风险自担, ~5,000h）
├── metadata/phase1_metrics.parquet     # 每 clip 一行全量 S0–S3 指标（打包结束时生成）
├── phase1_manifest_v1.parquet          # clip_id → 所在 shard 的映射（打包结束时生成）
└── README.md                           # dataset card
```

每个 shard 是 ~1GB 的 **WebDataset** tar：`{clip_id}.mp3|flac` + `{clip_id}.json`。
音频保留**原始采样率**（混采 24/32/44.1 kHz，训练前自行统一重采样）；`.flac` 是
S3 修剪后无损重编码的 clip（约 12%），其余为 Emilia 原始 mp3 字节。JSON 含
`text / duration / speaker / tier` 等字段。

> 打包是流式进行的：`data/*/shard-*.tar` 落盘一个可用一个（写入是 tmp+rename
> 原子操作，看到的 tar 一定是完整的）；两个 parquet 在最后才生成。

## 1. 读音频：用 `webdataset` 库（推荐，已实测）

注意**不要**用 `datasets.load_dataset("webdataset", ...)` 读音频 tar——shard 内
mp3/flac 混排，HF 的 builder 要求字段同质，会报
`ValueError: ... don't share the same types`。`webdataset` 库无此限制：

```python
import io, soundfile as sf
import webdataset as wds

ROOT = "/opt/dlami/nvme/leolxliu/emilia-zh-full/export/hf_phase1"

def decode_audio(key, data):
    """mp3/flac 统一解码为 (float32 ndarray, sr)。本机 torchaudio 缺 torchcodec
    后端, 不要用 wds.torch_audio, 用 soundfile 走字节流。"""
    if key.split(".")[-1].lower() not in ("mp3", "flac", "wav", "ogg"):
        return None
    return sf.read(io.BytesIO(data), dtype="float32")

def unify(sample):
    """屏蔽 mp3/flac 扩展名差异, 统一成 audio 字段。"""
    audio, sr = sample.get("mp3") or sample.get("flac")
    return {"id": sample["__key__"], "audio": audio, "sr": sr, "meta": sample["json"]}

ds = (wds.WebDataset(f"{ROOT}/data/prime/shard-{{00000..00007}}.tar",
                     shardshuffle=100)        # shard 级打散
      .shuffle(1000)                          # 样本级打散缓冲
      .decode(decode_audio)
      .map(unify))

for s in ds:
    print(s["id"], s["sr"], len(s["audio"]) / s["sr"], s["meta"]["text"][:20])
    break
```

shard 范围写法 `shard-{00000..00007}` 按需扩大；要用全部已落盘的 shard：

```python
import glob
urls = sorted(glob.glob(f"{ROOT}/data/prime/shard-*.tar"))
ds = wds.WebDataset(urls, shardshuffle=100)...
```

接 PyTorch 训练（`wds.WebLoader` 即 DataLoader 的 wds 封装）：

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
duckdb.sql(f"""
    SELECT shard, count(*) FROM '{ROOT}/phase1_manifest_v1.parquet'
    WHERE clip_id IN (SELECT clip_id FROM '{M}' WHERE aes_pq >= 7.5)
    GROUP BY 1 ORDER BY 1
""")
```

parquet 也可以用 HF datasets 正常加载（这个不受混排影响）：

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
喂模型前统一重采样（如 `soxr`/`librosa.resample`/`torchaudio.functional.resample`），
不要假设整库同采样率。
