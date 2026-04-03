"""Configuration and path management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ModelConfig:
    """Configuration for a single translation model."""

    name: str
    data_path: Path
    model_path: Path
    bpe_path: Path


@dataclass
class Config:
    """Central configuration for all translation models."""

    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    dictionaries: List[Path] = field(default_factory=list)
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    def __post_init__(self):
        if not self.dictionaries:
            dict_dir = self.data_dir / "dictionaries"
            self.dictionaries = [
                dict_dir / "dict.txt",
                dict_dir / "中国历史地名词典.txt",
                dict_dir / "古代人名（25w）.txt",
                dict_dir / "成语（5W）.txt",
            ]
        if not self.models:
            d = self.data_dir
            models_dir = d / "models"
            bpe_dir = d / "bpe"
            self.models = {
                "old2new": ModelConfig(
                    "old2new",
                    d / "dataflow_old2new",
                    models_dir / "model_old2new.pt",
                    bpe_dir / "old.bpe",
                ),
                "new2old": ModelConfig(
                    "new2old",
                    d / "dataflow_new2old",
                    models_dir / "model_new2old.pt",
                    bpe_dir / "new.bpe",
                ),
                "addpunc": ModelConfig(
                    "addpunc",
                    d / "dataflow_addpunc",
                    models_dir / "model_addpunc.pt",
                    bpe_dir / "addpunc.bpe",
                ),
                "genshici": ModelConfig(
                    "genshici",
                    d / "dataflow_shici",
                    models_dir / "model_shici.pt",
                    bpe_dir / "shici.bpe",
                ),
            }
