# Classic Chinese Translate

## Project Overview

Standalone classical Chinese translation engine using fairseq transformer models. Migrated from the WebTrans Django project.

## Environment

- Python 3.9, managed with **uv**
- torch 1.13.1, numpy < 1.24
- Custom fairseq 0.6.2 (vendored at `vendored/fairseq/`)
- Model/data files from HuggingFace (see Data Setup below)

## Data Setup

Before running the project, download model and data files from HuggingFace to the local `data/` folder:

```bash
# Install huggingface-hub if not already available
uv pip install huggingface-hub

# Download dataset (BPE codes, training data, dictionaries)
hf download bangboom/chinese-translation-data --local-dir data/dataset

# Download models (~5.2GB)
hf download bangboom/chinese-translation-models --local-dir data/model
```

### data/ folder structure

```
data/
├── dataset/                 # bangboom/chinese-translation-data
│   ├── bpe_file/            # BPE codes (old.bpe, new.bpe, addpunc.bpe, shici.bpe)
│   ├── dataflow_old2new/    # Binarized training data for old→new
│   ├── dataflow_new2old/    # Binarized training data for new→old
│   ├── dataflow_addpunc/    # Binarized training data for punctuation
│   ├── dataflow_shici/      # Binarized training data for poetry
│   └── dictionary/          # Lookup dictionaries
└── model/                   # bangboom/chinese-translation-models
    ├── model_old2new.pt
    ├── model_new2old.pt
    ├── model_addpunc.pt
    └── model_shici.pt
```

## Dependencies

```bash
uv sync
uv pip install -e ./vendored/fairseq
uv pip install huggingface-hub
```

## Usage

All commands go through `uv run`:

```bash
uv run cct old2new "袁州火，龙庆，延安，吉安，杭州，大都诸路属县水，民饥，赈粮有差。"
uv run cct new2new "现代文输入"
uv run cct add-punc "无标点古文"
uv run cct gen-poetry "白话文"
```

- `old2new` — Classical Chinese to Modern Chinese
- `new2old` — Modern Chinese to Classical Chinese
- `add-punc` — Add punctuation to unpunctuated classical Chinese
- `gen-poetry` — Generate poetry from modern Chinese text

## Architecture

```
src/classic_chinese_translate/
├── cli.py         # CLI entry point (uv run cct)
├── translator.py  # FairseqTranslator — model loading & inference
├── tokenizer.py   # ChineseTokenizer — jieba segmentation + BPE
├── bpe.py         # BPE encoding (Sennrich 2015)
└── config.py      # Path management for models/data/dictionaries

vendored/fairseq/  # Custom fairseq 0.6.2 fork
data/
├── dataset/        # BPE codes, training data, dictionaries (from HuggingFace)
└── model/          # fairseq .pt model files (from HuggingFace)
```

## Translation Pipeline

```
Input text → jieba segmentation → BPE tokenization → fairseq inference → detokenize
```

## Key Files (source project)

- Original code: `/Users/bangboom/Documents/WebTrans/webfortrans/my_code/`
- Original data: `/Users/bangboom/Documents/WebTrans/webfortrans/myfiles/`
- `load_test.py` → refactored into `translator.py`
- `try_load.py` → merged into `translator.py` as `FairseqTranslator` class
- `use_bpe.py` → cleaned into `bpe.py`
- `views.py` pipeline → `cli.py` + `tokenizer.py`

## Deferred (Phase 2)

- Dictionary/corpus lookup features from `ClassForWeb.py` (depends on Django ORM, needs DB data export)

## Notes

- Model loading takes ~10s per model, models are cached after first load
- torch 1.9.0 has no macOS ARM64 wheels; 1.13.1 is used and compatible
- The sys.argv hack from the original `load_test.py` has been replaced with `parse_args_and_arch(parser, input_args=...)`
