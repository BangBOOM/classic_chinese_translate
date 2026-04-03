"""Chinese text tokenizer: jieba segmentation + BPE encoding."""

import jieba


class ChineseTokenizer:
    """Handles jieba segmentation and BPE tokenization."""

    def __init__(self, bpe, dictionary_paths=None):
        self.bpe = bpe
        for path in dictionary_paths or []:
            jieba.load_userdict(str(path))

    def tokenize(self, src):
        """Full pipeline: strip spaces -> jieba cut -> BPE segment."""
        src = src.replace(" ", "")
        src_token = " ".join(jieba.cut(src, HMM=False))
        return self.bpe.segment(src_token)

    def detokenize(self, tgt):
        """Remove inter-token spaces from model output."""
        return tgt.replace(" ", "")
