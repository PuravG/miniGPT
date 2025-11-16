"""
Data loading and processing module for GPT training.

This module provides classes and functions for downloading, loading, and processing
text data for training language models.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, List
from urllib.request import urlretrieve

from config import DataConfig


class Vocabulary:
    """Handles character-level vocabulary encoding and decoding."""
    
    def __init__(self, text: str):
        """
        Initialize vocabulary from text.
        
        Args:
            text: Input text to build vocabulary from.
        """
        # Create sorted list of unique characters
        self.characters = sorted(list(set(text)))
        self.vocab_size = len(self.characters)
        
        # Create character to integer mapping
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.characters)}
        # Create integer to character mapping
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.characters)}
    
    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of integers.
        
        Args:
            text: Input string to encode.
            
        Returns:
            List of integers representing the encoded text.
        """
        return [self.stoi[c] for c in text]
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode a list of integers into a string.
        
        Args:
            indices: List of integers to decode.
            
        Returns:
            Decoded string.
        """
        return ''.join([self.itos[i] for i in indices])
    
    @property
    def encode_func(self) -> Callable[[str], List[int]]:
        """Return encode function for backward compatibility."""
        return self.encode
    
    @property
    def decode_func(self) -> Callable[[List[int]], str]:
        """Return decode function for backward compatibility."""
        return self.decode


class DataLoader:
    """Handles data downloading, loading, and batching."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: DataConfig instance with data settings.
        """
        self.config = config
        self.vocab: Optional[Vocabulary] = None
        self.train_data: Optional[torch.Tensor] = None
        self.val_data: Optional[torch.Tensor] = None
    
    def download_data(self) -> Optional[Path]:
        """
        Download the data file if it doesn't exist.
        
        Returns:
            Path to the data file if successful, None otherwise.
        """
        file_path = self.config.text_path_obj
        
        if file_path.exists():
            print(f"'{file_path}' exists.")
            return file_path
        else:
            print(f"'{file_path}' does not exist.")
            try:
                print(f"Downloading from {self.config.data_url}...")
                urlretrieve(self.config.data_url, str(file_path))
                print(f"Downloaded to {file_path}")
                return file_path
            except Exception as e:
                print(f"Error downloading file: {e}")
                return None
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, Vocabulary]:
        """
        Load and process the data file.
        
        Returns:
            Tuple of (train_data, val_data, vocabulary).
        """
        file_path = self.config.text_path_obj
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Read text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f'Length of characters in the text: {len(text)}')
        
        # Create vocabulary
        self.vocab = Vocabulary(text)
        print(f'Vocab size used in our model is {self.vocab.vocab_size}')
        
        # Encode text to tensor
        data = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        
        # Split data into train and validation
        n = int(self.config.train_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        return self.train_data, self.val_data, self.vocab
    
    def sample_batch(
        self,
        split: str,
        max_context: int,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch from the dataset.
        
        Args:
            split: Either 'train' or 'val'.
            max_context: Maximum context length.
            batch_size: Number of samples in the batch.
            
        Returns:
            Tuple of (input_tensor, target_tensor).
        """
        if self.train_data is None or self.val_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        data = self.train_data if split == "train" else self.val_data
        
        if len(data) <= max_context:
            raise ValueError(f"Data length ({len(data)}) must be greater than max_context ({max_context})")
        
        # Sample random starting indices
        ix = torch.randint(len(data) - max_context, (batch_size,))
        
        # Create input and target sequences
        x = torch.stack([data[i:i+max_context] for i in ix])
        y = torch.stack([data[i+1:i+1+max_context] for i in ix])
        
        return x, y


# Backward compatibility functions
def download_data(file_path: str) -> Optional[str]:
    """
    Download the Data to train the Model (backward compatibility).
    
    Args:
        file_path: Path to the data file.
        
    Returns:
        Path to the data file if successful, None otherwise.
    """
    from config import DataConfig
    config = DataConfig(text_path=file_path)
    loader = DataLoader(config)
    result = loader.download_data()
    return str(result) if result else None


def load_data(
    file_path: str,
    train_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, Callable[[str], List[int]], Callable[[List[int]], str]]:
    """
    Load the data from the file_path, process it, encode it and then return the train and val split.
    (backward compatibility)
    
    Args:
        file_path: Path to the data file.
        train_ratio: Ratio of data to use for training.
        
    Returns:
        Tuple of (train_data, val_data, encode_function, decode_function).
    """
    from config import DataConfig
    config = DataConfig(text_path=file_path, train_ratio=train_ratio)
    loader = DataLoader(config)
    train_data, val_data, vocab = loader.load_data()
    return train_data, val_data, vocab.encode_func, vocab.decode_func


def sample_batch(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    split: str,
    MAX_CONTEXT: int,
    BATCH_SIZE: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch from our dataset for training or validation (backward compatibility).
    
    Args:
        train_data: Training data tensor.
        val_data: Validation data tensor.
        split: Either 'train' or 'val'.
        MAX_CONTEXT: Maximum context length.
        BATCH_SIZE: Batch size.
        
    Returns:
        Tuple of (input_tensor, target_tensor).
    """
    from config import DataConfig
    config = DataConfig()
    loader = DataLoader(config)
    loader.train_data = train_data
    loader.val_data = val_data
    return loader.sample_batch(split, MAX_CONTEXT, BATCH_SIZE)
