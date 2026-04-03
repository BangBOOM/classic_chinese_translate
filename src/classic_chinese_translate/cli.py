"""CLI entry point for classical Chinese translation."""

import argparse

from .bpe import get_bpe_res
from .config import Config
from .tokenizer import ChineseTokenizer
from .translator import FairseqTranslator

# Module-level model cache to avoid reloading
_translators = {}
_tokenizers = {}


def _get_translator(name):
    if name not in _translators:
        cfg = Config().models[name]
        print(f"Loading model: {name}...")
        _translators[name] = FairseqTranslator(cfg.data_path, cfg.model_path)
    return _translators[name]


def _get_tokenizer(name):
    if name not in _tokenizers:
        cfg = Config()
        bpe = get_bpe_res(str(cfg.models[name].bpe_path))
        _tokenizers[name] = ChineseTokenizer(bpe, cfg.dictionaries)
    return _tokenizers[name]


def _translate(model_name, src):
    tokenizer = _get_tokenizer(model_name)
    translator = _get_translator(model_name)
    tokenized = tokenizer.tokenize(src)
    result = translator.translate([tokenized])
    return tokenizer.detokenize(result)


def _gen_poetry(src):
    cfg = Config()
    bpe = get_bpe_res(str(cfg.models["genshici"].bpe_path))
    src_clean = src.strip().replace(" ", " , ")
    tokenized = bpe.segment(src_clean)
    translator = _get_translator("genshici")
    result = translator.translate([tokenized]).replace(" ", "").replace("。", "。\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Classical Chinese Translation Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # old2new: classical Chinese to modern Chinese
    p = subparsers.add_parser("old2new", help="Translate classical Chinese to modern Chinese")
    p.add_argument("input", help="Input text in classical Chinese")

    # new2old: modern Chinese to classical Chinese
    p = subparsers.add_parser("new2old", help="Translate modern Chinese to classical Chinese")
    p.add_argument("input", help="Input text in modern Chinese")

    # add-punc: add punctuation to classical Chinese
    p = subparsers.add_parser("add-punc", help="Add punctuation to unpunctuated classical Chinese")
    p.add_argument("input", help="Input text without punctuation")

    # gen-poetry: poetry generation
    p = subparsers.add_parser("gen-poetry", help="Generate poetry from modern Chinese text")
    p.add_argument("input", help="Input text in modern Chinese")

    args = parser.parse_args()

    if args.command == "old2new":
        result = _translate("old2new", args.input)
    elif args.command == "new2old":
        result = _translate("new2old", args.input)
    elif args.command == "add-punc":
        result = _translate("addpunc", args.input)
    elif args.command == "gen-poetry":
        result = _gen_poetry(args.input)
    else:
        parser.print_help()
        return

    print(result)


if __name__ == "__main__":
    main()
