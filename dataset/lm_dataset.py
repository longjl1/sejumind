from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────────────────────
# 全局预处理 / 后处理工具函数
# ──────────────────────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Resolve relative paths against both CWD and project root for stable CLI behavior.
        resolved_data_path = data_path
        if isinstance(data_path, str):
            if os.path.exists(data_path):
                resolved_data_path = os.path.abspath(data_path)
            else:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                project_candidate = os.path.abspath(os.path.join(project_root, data_path))
                if os.path.exists(project_candidate):
                    resolved_data_path = project_candidate
                else:
                    raise FileNotFoundError(
                        f"Data file not found: '{data_path}'. Checked:\n"
                        f"- {os.path.abspath(data_path)}\n"
                        f"- {project_candidate}"
                    )

        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=resolved_data_path, split="train")

    def __len__(self):
        return len(self.samples)
    
    # 得到jsonl文件中的一行
    def __getitem__(self, idx):
        sample = self.samples[idx]

    # tokenizer编码
        tokens = self.tokenizer(
            str(sample["text"]),  # 假设jsonl文件中每行有一个"text"字段
            add_special_tokens=False,
            max_length=self.max_length - 2,  # 预留 BOS + EOS 的位置
            truncation=True,
        ).input_ids

        # 添加BOS和EOS标记 右侧用 PAD 补齐到 max_length
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens+[self.tokenizer.pad_token_id]*(self.max_length-len(tokens))  # padding到max_length
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_mask
