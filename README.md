# Classic Chinese Translate

古文翻译引擎 — 基于 fairseq transformer 模型的文言文/现代文互译、古文断句标点、诗词生成工具。

> 这是大学时期的项目，最早嵌套在一个Web项目，这次从中提取出来做成了一个最小化可运行的版本，作为纪念。

## 功能

- **古文→现代文** — 文言文翻译为现代汉语
- **现代文→古文** — 现代汉语翻译为文言文
- **古文断句** — 为无标点古文自动添加标点
- **诗词生成** — 根据白话文生成古诗词

## 安装

```bash
# 安装项目依赖
uv sync
uv pip install -e ./vendored/fairseq
uv pip install huggingface-hub

# 下载模型与数据
hf download bangboom/chinese-translation-data --local-dir data/dataset
hf download bangboom/chinese-translation-models --local-dir data/model
```

## 使用

### 古文翻译为现代文

```bash
$ uv run cct old2new "袁州火，龙庆，延安，吉安，杭州，大都诸路属县水，民饥，赈粮有差。"
```
> 袁州火灾、龙庆、延安、吉安、杭州、大都诸路属县水灾，人民饥荒，赈济粮食不等。

### 古文添加标点

```bash
$ uv run cct add-punc "黄帝者少典之子姓公孙名曰轩辕"
```
> 黄帝者，少典之子，姓公孙，名曰轩辕。

### 诗词生成

支持指定关键词和作者风格生成古诗词，也可以附加词牌名生成词。

```bash
$ uv run cct gen-poetry "月色 落花 凄凉 夜深 杜甫"
```
> 月色兼云白，秋声入夜长。  
> 落花知客意，啼鸟为人忙。  
> 迢遰他乡远，凄凉此夜长。  
> 夜深谁与语，欹枕听鸣榔。

```bash
$ uv run cct gen-poetry "月色 落花 凄凉 夜深 欧阳修 卜算子"
```
> 月色上帘栊，月色侵庭户。  
> 门掩落花春昼长，人静无人语。  
> 一枕独凄凉，万事都如许。  
> 谁道秋宵梦不成，却到夜深否。

## 项目结构

```
├── src/classic_chinese_translate/   # 主要代码
│   ├── translator.py                # fairseq 模型加载与推理
│   ├── tokenizer.py                 # jieba 分词 + BPE 编码
│   ├── bpe.py                       # BPE 子词编码
│   ├── config.py                    # 模型/数据路径配置
│   └── cli.py                       # 命令行入口
├── vendored/fairseq/                # 自定义 fairseq 0.6.2
└── data/
    ├── dataset/                     # BPE编码、训练数据、词典（HuggingFace）
    └── model/                       # fairseq 模型文件（HuggingFace）
```

## 翻译流程

```
输入文本 → jieba 分词 → BPE 子词编码 → fairseq 模型推理 → 去空格输出
```

## 许可

本项目代码仅供学习研究使用。
