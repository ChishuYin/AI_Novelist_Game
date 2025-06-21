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
