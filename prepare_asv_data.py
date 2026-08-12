import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import\
    soundfile as sf


ROOT = Path(__file__).resolve().parent

print(ROOT)

PARQUET_DIR = ROOT / "data" / "ASVspoof_2019_LA" / "data"
WAV_ROOT = ROOT / "data" / "ASVspoof_2019_LA" / "wav"

CONFIG = {
    "train": {
        "parquet": PARQUET_DIR / "train-00000-of-00001.parquet",
        "jsonl": ROOT / "data" / "asv_fake_audio.jsonl",
    },
    "validation": {
        "parquet": PARQUET_DIR / "validation-00000-of-00001.parquet",
        "jsonl": ROOT / "data" / "asv_validation_fake_audio.jsonl",
    },
    "test": {
        "parquet": PARQUET_DIR / "test-00000-of-00001.parquet",
        "jsonl": ROOT / "data" / "asv_test_fake_audio.jsonl",
    },
}


def convert_label(key):
    """
    ASVspoof2019 LA:
    bonafide / 0 -> Real.
    spoof    / 1 -> Fake.
    """
    if isinstance(key, str):
        key = key.strip().lower()

        if key in {"bonafide", "real", "genuine"}:
            return "Real."

        if key in {"spoof", "fake"}:
            return "Fake."

    if key == 0:
        return "Real."

    if key == 1:
        return "Fake."

    raise ValueError(f"无法识别标签: {key}")


def get_audio_bytes(audio):
    if isinstance(audio, dict):
        data = audio.get("bytes")
        if data is not None:
            return data

        path = audio.get("path")
        if path and Path(path).exists():
            return Path(path).read_bytes()

    if isinstance(audio, (bytes, bytearray)):
        return audio

    raise ValueError(f"无法解析 audio 字段: {type(audio)}")


def convert_split(split, parquet_path, jsonl_path):
    wav_dir = WAV_ROOT / split
    wav_dir.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(parquet_path)

    total = parquet.metadata.num_rows
    count = 0

    print(f"\n开始处理 {split}: {total} 条")

    with open(jsonl_path, "w", encoding="utf-8") as fout:

        # 分批处理，避免一次把整个 parquet 加载进内存
        for batch in parquet.iter_batches(batch_size=256):

            rows = batch.to_pylist()

            for row in rows:
                audio_name = str(row["audio_file_name"])
                audio_name = Path(audio_name).stem

                wav_path = wav_dir / f"{audio_name}.wav"

                audio_bytes = get_audio_bytes(row["audio"])

                waveform, sample_rate = sf.read(
                    io.BytesIO(audio_bytes),
                    dtype="float32"
                )

                sf.write(
                    wav_path,
                    waveform,
                    sample_rate,
                    subtype="PCM_16"
                )

                answer = convert_label(row["key"])

                item = {
                    "messages": [
                        {
                            "role": "user",
                            "audio": str(wav_path.resolve()),
                            "content": "Is this audio real or fake?"
                        },
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    ]
                }

                fout.write(
                    json.dumps(item, ensure_ascii=False) + "\n"
                )

                count += 1

                if count % 1000 == 0 or count == total:
                    print(f"{split}: {count}/{total}")

    print(f"{split} 完成")
    print(f"JSONL: {jsonl_path}")


def main():
    for split, cfg in CONFIG.items():

        parquet_path = cfg["parquet"]

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"找不到文件: {parquet_path}"
            )

        convert_split(
            split,
            parquet_path,
            cfg["jsonl"]
        )

    print("\n全部转换完成。")


if __name__ == "__main__":
    main()