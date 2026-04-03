# Classic Chinese Translate

古文翻译引擎 — 基于 fairseq transformer 模型的文言文/现代文互译、古文断句标点、诗词生成工具。

## 功能

- **古文→现代文** — 文言文翻译为现代汉语
- **现代文→古文** — 现代汉语翻译为文言文
- **古文断句** — 为无标点古文自动添加标点
- **诗词生成** — 根据白话文生成古诗词

## 安装

```bash
# 安装项目依赖
uv sync
# 安装自定义 fairseq
uv pip install -e ./vendored/fairseq
```

## 使用

```bash
# 古文翻译为现代文
uv run cct old2new "袁州火，龙庆，延安，吉安，杭州，大都诸路属县水，民饥，赈粮有差。"

# 现代文翻译为古文
uv run cct new2old "现代文输入"

# 古文添加标点
uv run cct add-punc "黄帝者少典之子姓公孙名曰轩辕"

# 诗词生成
uv run cct gen-poetry "白话文输入"
```

## 项目结构

```
├── src/classic_chinese_translate/   # 主要代码
│   ├── translator.py                # fairseq 模型加载与推理
│   ├── tokenizer.py                 # jieba 分词 + BPE 编码
│   ├── bpe.py                       # BPE 子词编码
│   ├── config.py                    # 模型/数据路径配置
│   └── cli.py                       # 命令行入口
├── vendored/fairseq/                # 自定义 fairseq 0.6.2
└── data/                            # 模型与数据文件（符号链接）
```

## 翻译流程

```
输入文本 → jieba 分词 → BPE 子词编码 → fairseq 模型推理 → 去空格输出
```

## 许可

本项目代码仅供学习研究使用。
