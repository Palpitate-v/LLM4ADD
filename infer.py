import gc
import json
import os
import pathlib
from typing import Any, Dict, List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import transformers
from accelerate import cpu_offload
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from batch_modeling_qwen import QWenLMHeadModel
from tokenization_qwen import QWenTokenizer

torch.manual_seed(1234)

# =========================
# 需要根据实际情况修改的配置
# =========================
PROJECT_DIR = pathlib.Path(__file__).resolve().parent

# 必须与训练时使用的基础模型保持一致。
MODEL_NAME_OR_PATH = r"F:\Workplace\PycharmWorkplace\ALLM4ADD\models\models\qwen--Qwen-Audio-Chat\snapshots\master"

# 训练输出结构：output/temp/checkpoint-15865/
CHECKPOINT_ROOT = PROJECT_DIR / "output" / "temp"

# 默认测试最终 checkpoint。若想测试所有 checkpoint，改为 None。
CHECKPOINT_NAME = "checkpoint-15865"

TEST_DATA_PATH = PROJECT_DIR / "data" / "asv_test_fake_audio.jsonl"
RESULT_DIR = PROJECT_DIR / "infer_eer"
# RTX 5060 显存有限，基础模型会部分卸载到磁盘；使用 batch size 1 更稳妥。
BATCH_SIZE = 1
# 用于保存被 Accelerate 卸载到 CPU/磁盘的模型参数。
OFFLOAD_DIR = PROJECT_DIR / "offload"


def get_checkpoint_folders() -> List[pathlib.Path]:
    """获取需要测试的 LoRA checkpoint。"""
    if CHECKPOINT_NAME is not None:
        checkpoint = CHECKPOINT_ROOT / CHECKPOINT_NAME
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"找不到 checkpoint：{checkpoint}\n"
                f"请检查 CHECKPOINT_NAME 或 CHECKPOINT_ROOT 是否正确。"
            )
        return [checkpoint]

    checkpoints = [
        path for path in CHECKPOINT_ROOT.glob("checkpoint-*") if path.is_dir()
    ]
    checkpoints.sort(
        key=lambda path: int(path.name.split("-")[-1])
        if path.name.split("-")[-1].isdigit()
        else -1
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"在 {CHECKPOINT_ROOT} 下没有找到 checkpoint-* 目录。"
        )
    return checkpoints


def from_list_format(list_format: List[Dict[str, str]]) -> str:
    """将音频和文本拼接成 Qwen-Audio 使用的输入格式。"""
    text = ""
    num_audios = 0
    audio_start_tag, audio_end_tag = "<audio>", "</audio>"

    for element in list_format:
        if "audio" in element:
            num_audios += 1
            text += f"Audio {num_audios}:"
            text += audio_start_tag + element["audio"] + audio_end_tag
            text += "\n"
        elif "text" in element:
            text += element["text"]
        else:
            raise ValueError(f"Unsupported element: {element}")
    return text


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    system: str = "",
):
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def tokenize_text(role: str, content: str):
        audio_info = tokenizer.process_audio(content)
        tokens = tokenizer.encode(
            role,
            allowed_special=set(tokenizer.AUDIO_ST),
            audio_info=audio_info,
        )
        tokens += nl_tokens
        tokens += tokenizer.encode(
            content,
            allowed_special=set(tokenizer.AUDIO_ST),
            audio_info=audio_info,
        )
        return f"{role}\n{content}", tokens

    system_text, system_tokens_part = tokenize_text("system", system)
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

    context_tokens = system_tokens
    raw_text = f"{im_start}{system_text}{im_end}"

    _, user_tokens = tokenize_text("user", query)
    context_tokens += nl_tokens + im_start_tokens + user_tokens
    context_tokens += im_end_tokens + nl_tokens
    context_tokens += im_start_tokens + tokenizer.encode("assistant") + nl_tokens

    raw_text += (
        f"\n{im_start}user\n{query}{im_end}"
        f"\n{im_start}assistant\n"
    )
    audio_info = tokenizer.process_audio(raw_text)
    return raw_text, context_tokens, audio_info


def preprocess(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = 2000,
    prompt_message: str = "Can you determine if this audio is fake or real?",
    system_message: str = "You are a helpful assistant.",
) -> Dict[str, Any]:
    audio_path = source[0]["audio"]
    query_message = source[0].get("content") or prompt_message

    query = tokenizer.from_list_format(
        [{"audio": audio_path}, {"text": query_message}]
    )
    _, context_tokens, audio_info = make_context(
        tokenizer,
        query,
        system=system_message,
    )

    # 推理时不能截断音频相关 token，因此保留原始长度。
    input_ids = torch.tensor(context_tokens, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id),
        "audio_info": audio_info,
    }


class LazySupervisedDataset(Dataset):
    """测试数据集，按需处理音频。"""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int = 2000,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index not in self.cached_data_dict:
            self.cached_data_dict[index] = preprocess(
                self.raw_data[index]["messages"],
                self.tokenizer,
                self.max_len,
            )
        return self.cached_data_dict[index]


class CustomDataCollator(transformers.DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 不直接 pop 原始 feature，避免缓存中的样本被破坏。
        audio_infos = [feature.get("audio_info") for feature in features]
        text_features = [
            {key: value for key, value in feature.items() if key != "audio_info"}
            for feature in features
        ]
        batch = super().__call__(text_features)

        if any(audio_infos):
            audio_span_tokens = []
            audio_name_files = []
            valid_audio_infos = [info for info in audio_infos if info is not None]

            for info in valid_audio_infos:
                audio_span_tokens.extend(info["audio_span_tokens"])
                audio_name_files.extend(info["audio_urls"])

            batch["audio_info"] = {
                "input_audios": torch.concat(
                    [info["input_audios"] for info in valid_audio_infos]
                ),
                "audio_span_tokens": audio_span_tokens,
                "input_audio_lengths": torch.concat(
                    [info["input_audio_lengths"] for info in valid_audio_infos]
                ),
                "input_audio_names": audio_name_files,
            }
        return batch


def load_non_lora_trainables(
    merged_model: torch.nn.Module,
    checkpoint_path: pathlib.Path,
) -> None:
    """加载训练音频编码器时保存的非 LoRA 参数。"""
    non_lora_path = checkpoint_path / "non_lora_trainables.bin"
    if not non_lora_path.exists():
        print(f"提示：{non_lora_path} 不存在，跳过非 LoRA 参数加载。")
        return

    print(f"正在加载非 LoRA 参数：{non_lora_path}")
    non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
    non_lora_trainables = {
        (key[11:] if key.startswith("base_model.") else key): value
        for key, value in non_lora_trainables.items()
    }
    non_lora_trainables = {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in non_lora_trainables.items()
    }

    missing_keys, unexpected_keys = merged_model.load_state_dict(
        non_lora_trainables,
        strict=False,
    )
    print(
        "非 LoRA 参数加载完成："
        f"missing_keys={len(missing_keys)}, "
        f"unexpected_keys={len(unexpected_keys)}"
    )


def get_single_token_id(tokenizer: PreTrainedTokenizer, text: str) -> int:
    """获取 Real/Fake 的单 token ID。"""
    token_ids = tokenizer.encode(text)
    if len(token_ids) != 1:
        raise RuntimeError(
            f"'{text}' 被编码成了多个 token：{token_ids}，"
            "请检查 tokenizer 或模型是否与训练时一致。"
        )
    return token_ids[0]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前 PyTorch 没有检测到 CUDA。该 Qwen-Audio 模型通常需要 GPU 推理。"
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"找不到测试数据：{TEST_DATA_PATH}")

    checkpoint_folders = get_checkpoint_folders()
    print("待测试 checkpoint：")
    for checkpoint in checkpoint_folders:
        print(f"  {checkpoint}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"正在加载 tokenizer：{MODEL_NAME_OR_PATH}")
    tokenizer = QWenTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.padding_side = "left"

    test_data = []
    with TEST_DATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                test_data.append(json.loads(line))

    test_data = test_data[:10]
    print(f"测试样本数：{len(test_data)}")
    test_dataset = LazySupervisedDataset(test_data, tokenizer=tokenizer)
    data_collator = CustomDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    fake_token_id = get_single_token_id(tokenizer, "Fake")
    real_token_id = get_single_token_id(tokenizer, "Real")
    print(f"Fake token id: {fake_token_id}, Real token id: {real_token_id}")

    device = torch.device("cuda:0")

    for checkpoint_path in checkpoint_folders:
        output_path = RESULT_DIR / f"{checkpoint_path.name}.json"
        if output_path.exists():
            print(f"结果已存在，跳过：{output_path}")
            continue

        print(f"\n正在加载 checkpoint：{checkpoint_path}")
        # 先把基础模型完整放在 CPU 内存中，避免自动 device map 提前触发磁盘分配。
        base_model = QWenLMHeadModel.from_pretrained(
            MODEL_NAME_OR_PATH,
            device_map={"": "cpu"},
            # Windows 下避免 CPU bfloat16/FlashAttention 导致原生层崩溃。
            fp16=True,
            use_flash_attn=False,
            offload_state_dict=True,
            trust_remote_code=True,
        ).eval()

        # PEFT 0.8.2 会根据 hf_device_map 再次自动 dispatch；
        # 这里先清除映射，让 LoRA 保持在 CPU 上加载和合并。
        if hasattr(base_model, "hf_device_map"):
            base_model.hf_device_map = None

        peft_model = PeftModel.from_pretrained(
            base_model,
            str(checkpoint_path),
        )
        merged_model = peft_model.merge_and_unload().eval()

        # finetune_audio.py 训练了音频编码器，因此必须加载这个文件。
        load_non_lora_trainables(merged_model, checkpoint_path)

        # AudioEncoder 使用自定义的 encode() 调用路径。若把它留在模型
        # 注册树中，cpu_offload 会给它加 hook 并把参数变成 meta tensor，
        # 随后第一次处理音频就会报 Cannot copy out of meta tensor。
        # 暂时摘出它，让 Accelerate 只卸载语言模型；之后再固定到 GPU。
        audio_encoder = merged_model.transformer.audio
        merged_model.transformer.audio = None

        # 模型权重保留在 CPU 内存，计算层按需临时搬到 GPU。
        merged_model = cpu_offload(
            merged_model,
            execution_device=device,
        )
        audio_encoder.to(device)
        merged_model.transformer.audio = audio_encoder
        merged_model.eval()

        answers = {}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=checkpoint_path.name):
                audio_names = batch["audio_info"]["input_audio_names"]
                logits = merged_model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    audio_info=batch["audio_info"],
                ).logits

                last_logits = logits[:, -1, :]
                target_logits = last_logits[:, [fake_token_id, real_token_id]]

                for index, audio_name in enumerate(audio_names):
                    answers[audio_name] = [
                        float(value) for value in target_logits[index].cpu()
                    ]

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(answers, file, indent=4, ensure_ascii=False)

        print(f"测试完成，结果已保存：{output_path}")

        del merged_model
        del peft_model
        del base_model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
