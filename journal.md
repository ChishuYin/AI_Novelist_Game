# 📓 Project Log: Fine-Tuning GPT2 for Jin Yong–Style Chinese Novel Generation

---

## Day 1: Project Initialization

### Objective
To develop a generative language model capable of producing original Chinese fiction in the narrative and stylistic tradition of Jin Yong (金庸), using reference texts including:

- *The Legend of the Condor Heroes* (《射雕英雄传》)
- *Demi-Gods and Semi-Devils* (《天龙八部》)
- *The Smiling, Proud Wanderer* (《笑傲江湖》)
- *The Deer and the Cauldron* (《鹿鼎记》)

These texts were selected for their rich narrative structures, distinctive prose, philosophical depth, and varying degrees of genre subversion.

### Progress
- Local Python development environment set up
- HuggingFace account registered
- Defined stylistic and narrative objectives for output

---

## Day 2: Data Collection

### Corpus Acquired
Downloaded the following works:

- *The Legend of the Condor Heroes*
- *The Heaven Sword and Dragon Saber*
- *The Deer and the Cauldron*
- *The Smiling, Proud Wanderer*
- *Demi-Gods and Semi-Devils*

### Preprocessing
- Split each novel into parts no larger than 2MB
- Removed inline line breaks and extraneous whitespace

### Next Step
Build a structured dataset using HuggingFace `datasets.Dataset`.

---

## Day 3: Dataset Construction

### Progress
- Segmented texts into paragraphs
- Built structured HuggingFace dataset and saved locally

### Technical Details
- Custom text-splitting function implemented to preserve sentence logic
- Data volume: ~xxx paragraphs (est.)
- Average characters per paragraph: ~xxx

### Planned Tasks
- Prepare training arguments and configuration for model fine-tuning

---

## Day 4: Fine-Tuning Setup

### Tasks Completed
- Selected base model: `uer/gpt2-chinese-cluecorpussmall`
- Initialized and saved tokenizer (`tokenizer_gpt2_chinese/`)
- Created training script with PEFT + LoRA (`train_gpt2_chinese_lora.py`)
- Configured LoRA parameters (`r=8, alpha=16`, targeting `c_attn`)

### Technical Notes
- Successfully tokenized dataset with max_length=512
- Training pipeline structured using HuggingFace `Trainer`

---

## Day 5: Local Fine-Tuning Execution

### Objective
Fine-tune the small GPT2 Chinese model on local GPU with LoRA.

### Completed Tasks
- Ran 3 epochs of training using structured dataset
- Saved model artifacts to `outputs/gpt2_chinese_lora_small/`
- Output includes `adapter_model.safetensors`, `adapter_config.json`

### Highlights
- No quantization or bitsandbytes needed (GTX 1660 Super)
- LoRA significantly reduced memory usage and training time

### Next Step
- Write inference script and begin qualitative evaluation

---

## Day 6: Inference & Evaluation

### Task
Evaluate stylistic performance of the fine-tuned model through prompt-based inference.

### Output Prompt
“她站在桥头，望着雨雾中的城墙，心里忽然泛起一种奇怪的情绪。”

### Subjective Evaluation

| Metric               | Score (/5) |
|----------------------|------------|
| Word Choice          | 3.5        |
| Sentence Fluency     | 4.0        |
| Narrative Coherence  | 2.0        |
| Style Imitation      | 3.5        |

### Observations
- The model demonstrated stylistic fluency and lexical similarity to source texts
- Lacked coherence and story continuity in longer generations
- Larger-scale model (e.g., DeepSeek-Coder) will be considered next

---

## Next Phase
- Upgrade model to `deepseek-ai/DeepSeek-Coder-6.7B-base`
- Explore cloud-based training (e.g., Colab Pro, Kaggle, or rented GPU)
- Conduct BLEU/ROUGE-style quantitative evaluation
- Refine prompt engineering and implement beam/nucleus decoding


