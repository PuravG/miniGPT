"""Configuration classes for GPT model, training, and data handling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the GPT model architecture."""
    
    vocab_size: int = 65
    max_context: int = 256
    embed_dim: int = 384
    num_heads: int = 6
    num_blocks: int = 6
    dropout: float = 0.2
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_context <= 0:
            raise ValueError("max_context must be positive")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")


@dataclass
class TrainingConfig:
    """Configuration for training the GPT model."""
    
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 5000
    batch_size: int = 64
    validation_freq: int = 100
    eval_iters: int = 200
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.validation_freq <= 0:
            raise ValueError("validation_freq must be positive")
        if self.eval_iters <= 0:
            raise ValueError("eval_iters must be positive")


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    text_path: str = "shakespeare.txt"
    train_ratio: float = 0.9
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be in (0, 1)")
    
    @property
    def text_path_obj(self) -> Path:
        """Return text_path as a Path object."""
        return Path(self.text_path)

