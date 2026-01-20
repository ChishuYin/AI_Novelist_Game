# gpt2-finetune-novel
Finetuned GPT2 model on custom novel dataset
# GPT2-Chinese Novel Generator (LoRA Fine-Tuned)

This project fine-tunes a Chinese GPT2 model using a custom dataset of Chinese fiction, leveraging LoRA (Low-Rank Adaptation) for efficient training. The final model is capable of generating creative novel-style text in Chinese.

## 🔧 Model Configuration

- **Base model**: [`uer/gpt2-chinese-cluecorpussmall`](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)
- **LoRA adapter path**: `outputs/gpt2_chinese_lora_small`
- **Tokenizer**: Loaded from the LoRA path, with `use_fast=False` to avoid Chinese tokenization issues.

## 🏗️ How It Works

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("outputs/gpt2_chinese_lora_small", use_fast=False)

# Load base model + LoRA adapter
base_model = "uer/gpt2-chinese-cluecorpussmall"
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, "outputs/gpt2_chinese_lora_small")
model.eval()

# Text generation function
def generate(text, max_tokens=50):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=0.99,
            temperature=0.95
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    prompt = "她站在桥头，望着城墙，心里感到奇怪。"
    output = generate(prompt)
    print("📜 Output:\n", output)

# Output
📜 生成内容：
 她 站 在 桥 头 ， 望 着 雨 雾 中 的 城 墙 ， 心 里 忽 然 泛 起 一 种 奇 怪 的 情 绪 。 上 官 婉 儿 是 如 何 回 转 着 走 来 的 ？ 是 谁 已 经 走 了 多 少 步 ？ 他 在 桥 上 又 是 走 了 多 少 步 ？ 那 些 人 都 站 在 哪 里 ？ 他 不 再 想 走

# Project structure
gpt2_novelist/
├── outputs/                      # Contains fine-tuned LoRA model
├── generate.py                  # Inference script
├── train.py                     # (Optional) Training script
├── data/                        # Custom dataset (not uploaded)
└── README.md

# Requirements
pip install torch transformers peft

# Features
Supports Chinese text generation
LoRA for parameter-efficient fine-tuning
Sampling configuration with top_p and temperature for creative outputs

# TODO
Upload training script and dataset description
