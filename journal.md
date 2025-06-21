# Day 1：游戏启动！

## 我的目标
我要做一个能写出【金庸风格】小说的AI，参考的作品有：

- 《射雕英雄传》：英雄成长叙事，热血励志，少年成侠之路；爱国情怀浓厚，背景融入南宋抗金历史；正邪分明，人物形象鲜明（郭靖的忠厚、黄蓉的聪慧）；情节紧凑，兼具家仇国恨与儿女情长；文风明快大气，充满理想主义
- 《天龙八部》：结构宏大，多线交织，人物众多且性格复杂；探讨“人性本恶”、“佛性与魔性”的哲学命题；情节起伏跌宕，命运沉重，带有强烈宿命感
- 《笑傲江湖》：讽喻现实，主题深刻（权力斗争、自由 vs. 规训）；主角孤傲洒脱，反体制，情感复杂；文风兼具幽默与哲理，语言华美而富有文采
- 《鹿鼎记》：完全反类型化，主角“非侠”却纵横天下；讽刺权力、荒诞幽默，颠覆传统武侠结构；语言诙谐、对白机智，现实主义色彩浓厚

## 今日进度
完成了开发环境搭建，注册了HuggingFace账号。

## 想象中的AI
我希望它能接过我写的小说大纲，自动帮我写出精彩的段落。就像我脑海里的“替身写手”。

## 今日奖励
称号：造梦之初
技能点 +1


# Day 1: Game Start!

## My Goal  
I want to build an AI that can write novels in the **style of Jin Yong**. The reference works include:

- *The Legend of the Condor Heroes*: A classic hero’s journey with passionate and inspiring growth; strong patriotic themes set in the context of the Southern Song Dynasty's resistance against the Jin invaders; clear distinction between good and evil, with vivid characters (e.g., Guo Jing’s honesty and Huang Rong’s cleverness); tightly woven plot combining national crisis and personal emotions; energetic and idealistic writing style.

- *Demi-Gods and Semi-Devils*: Grand structure with interwoven storylines and a vast cast of complex characters; explores philosophical themes like the duality of human nature and the tension between Buddhist compassion and inner darkness; dramatic ups and downs with a heavy sense of fate.

- *The Smiling, Proud Wanderer*: A sharp satire on reality with deep themes (power struggles, freedom vs. constraint); a free-spirited and emotionally layered protagonist who resists the system; combines humor and philosophy with elegant, poetic prose.

- *The Deer and the Cauldron*: A complete genre subversion—an anti-hero protagonist navigating the world with wit; sharp satire on power, absurd humor, and a break from traditional wuxia tropes; witty dialogues, humorous narration, and strong realism.

## Progress Today  
Completed development environment setup and registered a HuggingFace account.

## My Dream AI  
I hope it can take the outlines I write and turn them into vivid, captivating passages—like a “ghostwriter in my mind.”

## Today’s Reward  
**Title**: *The Dream Begins*  
**Skill Point**: +1










# Day 2：语料猎人

## 我获取了这些小说：
- 《射雕英雄传》：网盘下载
- 《倚天屠龙记》：网盘下载
- 《鹿鼎记》：网盘下载
- 《笑傲江湖》：网盘下载
- 《天龙八部》：网盘下载

## 清洗方式
- 每部小说都分成上下两部（尽量控制在2mb以下）
- 去掉了句子中间的换行符和多余空格

## 明日计划
构建结构化数据集，用 Dataset 加载这些语料！

## 今日称号
📖 文本采集者

## 今日奖励
技能点 +1


# Day 2: Corpus Hunter

## 📚 Novels I Collected
- *The Legend of the Condor Heroes* – cloud download  
- *The Heaven Sword and Dragon Saber* – cloud download  
- *The Deer and the Cauldron* – cloud download  
- *The Smiling, Proud Wanderer* – cloud download  
- *Demi-Gods and Semi-Devils* – cloud download  

## 🧼 Cleaning Methods
- Each novel was split into **two parts**, keeping each file **under ~2MB**
- Removed **line breaks inside sentences** and cleaned **extra whitespace**

## 🔮 Tomorrow’s Plan
Build a **structured dataset** and load the corpora using `datasets.Dataset` from HuggingFace.

## 🏷️ Title of the Day
**📖 The Text Collector**

## 🎁 Today’s Reward
**+1 Skill Point**










# Day 3：结构祭坛

## 今日进展
- 成功将小说文本分段，构建出结构化 Dataset
- 数据量：约 xxx 个段落
- 每段平均字数：xxx

## 技术细节
- 使用自定义函数将文本按逻辑句子切块
- 使用 HuggingFace Dataset 保存为本地磁盘格式

## 明日目标
开始准备微调模型，定义训练参数

## 今日称号
🧪 数据炼金术士

## 今日奖励
技能点 +1


# Day 3: The Structure Altar

## Progress Today
- Successfully segmented the novel text into paragraphs and constructed a structured Dataset
- Data volume: approximately xxx paragraphs  
- Average characters per paragraph: xxx

## Technical Details
- Used a custom function to split the text into logical sentence chunks  
- Saved using HuggingFace `Dataset` in local disk format

## Goals for Tomorrow
Begin preparing for model fine-tuning and define training parameters

## Title of the Day
🧪 Data Alchemist

## Daily Reward
+1 Skill Point










# 📅 Day 4：模型驯养之门（轻量版）

## 🎯 今日任务
为中文 GPT2 小模型微调做好准备工作，构建 tokenizer、数据处理函数与训练脚本框架。

## ✅ 今日完成事项
- 成功选择并加载了 HuggingFace 中文 GPT2 小模型
- 使用 AutoTokenizer 成功解析数据（max_length=512）
- 搭建了本地 LoRA 训练脚本：train_gpt2_chinese_lora.py
- 使用 PEFT 和 LoRA 参数配置完成训练框架
- 将 tokenizer 保存到本地目录：`tokenizer_gpt2_chinese/`

## 🔧 技术要点
- 学会使用 tokenizer 与 dataset 结合进行批量 tokenize
- 熟悉训练脚本所需结构（Trainer + Dataset + LoRA）

## 🎁 今日称号
🎎 文风练习生

## 🧠 技能点 +2
- 掌握 tokenizer 与模型的基本协作流程
- 熟悉 LoRA 微调流程与模块结构


# 📅 Day 4: Model Taming Begins (Lightweight Version)

## 🎯 Goal
Prepare for fine-tuning a small Chinese GPT2 model with PEFT/LoRA, including tokenizer setup and training script construction.

## ✅ Tasks Completed
- Selected pretrained model: `uer/gpt2-chinese-cluecorpussmall`
- Loaded tokenizer using `AutoTokenizer`, configured for 512 tokens
- Constructed LoRA training script `train_gpt2_chinese_lora.py`
- Configured LoRA parameters and PEFT integration
- Saved tokenizer locally to `tokenizer_gpt2_chinese/`

## 🔧 Technical Highlights
- Learned to batch tokenize with HuggingFace tokenizer + datasets
- Built a modular training pipeline using Trainer + PEFT + LoRA

## 🎁 Title Earned
🎎 Stylist Apprentice

## 🧠 +2 Skill Points
- Mastered tokenizer/model interaction
- Learned the LoRA micro-finetuning setup











# 📅 Day 5：小说风格微调训练（本地）

## 🎯 今日任务
使用 uer/gpt2-chinese-cluecorpussmall 模型，在本地 GPU 上进行小说文本的微调训练，采用 LoRA 技术进行轻量参数调优。

## ✅ 今日完成事项
- 成功加载 HuggingFace 中文 GPT2 小模型
- 使用 HuggingFace Dataset 加载格式化小说数据
- 配置 LoRA 参数（r=8, alpha=16, target_modules=["c_attn"]）
- 在 GTX 1660 Super 上成功训练 3 个 epoch
- 模型保存路径：`outputs/gpt2_chinese_lora_small/`
- 输出文件包括 `adapter_model.safetensors` 与 `adapter_config.json`

## 🔧 技术要点
- 使用 PEFT + LoRA 训练大大降低了显存需求
- 训练无需使用 bitsandbytes 或 GPU 量化
- 输出模型需要与 base_model 一起加载才能使用

## 🎁 今日称号
🧙‍♀️ 文风塑形者

## 🧠 技能点 +2
- 熟练掌握 tokenizer + 数据切分 + LoRA 训练流程
- 掌握 HuggingFace 微调结构与保存方式

## 📌 下一步计划
- 编写推理测试脚本，输入小说提示语并生成风格化续写
- 对模型输出进行主观风格评分

# 📅 Day 5: Fine-tuning for Novel Style (Local Training)

## 🎯 Today’s Goal
Fine-tune the pretrained model `uer/gpt2-chinese-cluecorpussmall` on cleaned novel datasets using LoRA on local GPU (GTX 1660 Super).

## ✅ Tasks Completed
- Loaded pretrained GPT2 Chinese small model via HuggingFace
- Loaded preprocessed dataset from disk using `datasets`
- Configured LoRA (r=8, alpha=16, target_modules=["c_attn"])
- Trained for 3 epochs successfully on local GPU
- Model saved to `outputs/gpt2_chinese_lora_small/`
- Output includes `adapter_model.safetensors` and `adapter_config.json`

## 🔧 Key Technical Notes
- LoRA enables low-resource fine-tuning
- No bitsandbytes or GPU quantization needed
- Inference requires loading base model + adapter

## 🎁 New Title
🧙‍♀️ Stylist of Sentences

## 🧠 +2 Skill Points
- Mastered tokenizer usage, dataset preprocessing, LoRA training
- Learned HuggingFace + PEFT save/load pipeline

## 📌 Next Step
- Write an inference script to test style continuation
- Rate model output for stylistic accuracy










# 📅 Day 6：风格觉醒测试

## 🎯 今日任务
使用微调后的 GPT2 中文小模型生成小说段落，进行主观风格评估，识别模型能力与局限。

## ✅ 任务完成
- 成功运行推理脚本 `infer_gpt2_chinese_lora.py`
- 使用 Prompt：“她站在桥头，望着雨雾中的城墙，心里忽然泛起一种奇怪的情绪。”
- 模型成功生成 2~3 段落，语感初具古风，但存在重复与跳跃问题

## 🔍 风格评分（主观）
- 用词自然度：3.5/5
- 句法通顺：4/5
- 情节连贯性：2/5
- 风格模仿性：3.5/5

## 🧠 认知提升
小模型在风格仿写方面已有所成，但句子层面漂移明显，难以生成稳定的剧情段落 → 准备升级到大模型（DeepSeek）

## 🎁 今日称号
📝 风格观察者

## 🧠 技能点 +1
- 掌握微调模型评估方法
- 初步了解 prompt → 输出效果的影响因素


# 📅 Day 6: Style Awakening Test

## 🎯 Today's Task
Use the fine-tuned GPT2 Chinese small model to generate novel passages, and evaluate the stylistic quality of the outputs.

## ✅ Completed Tasks
- Ran the inference script `infer_gpt2_chinese_lora.py`
- Prompt used: “She stood on the bridge, staring at the fog-covered city wall…”
- Output was grammatically correct and stylistically similar to training data, but suffered from repetition and incoherence.

## 🔍 Subjective Style Evaluation
- Word Choice Naturalness: 3.5/5  
- Sentence Flow: 4/5  
- Narrative Coherence: 2/5  
- Style Mimicry: 3.5/5  

## 🧠 Key Insight
Small models can mimic style, but struggle with continuity and structural control. Prepare to upgrade to DeepSeek for larger-scale generation.

## 🎁 Title Unlocked
📝 Style Observer

## 🧠 +1 Skill Point
- Learned how to evaluate fine-tuned model behavior
- Understood the impact of prompt structure on model output
