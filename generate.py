"""Text generation script for GPT model."""

import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from model import GPT
from data import DataLoader
from config import ModelConfig, DataConfig


class Generator:
    """Handles text generation using a trained GPT model."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        device: str,
        model_path: str = "GPT_model.pt"
    ):
        """
        Initialize the Generator.
        
        Args:
            model_config: ModelConfig instance with model hyperparameters.
            data_config: DataConfig instance with data settings.
            device: Device to run generation on ('cuda', 'mps', or 'cpu').
            model_path: Path to the trained model checkpoint.
        """
        self.model_config = model_config
        self.data_config = data_config
        self.device = device
        self.model_path = Path(model_path)
        
        # Initialize data loader
        self.data_loader = DataLoader(data_config)
        self.vocab = None
        
        # Model will be initialized after vocabulary is loaded to get correct vocab_size
        self.model = None
    
    def load_vocabulary(self) -> None:
        """Load vocabulary from data file."""
        # Download data if needed
        data_path = self.data_loader.download_data()
        if not data_path:
            raise RuntimeError("Data download was unsuccessful")
        
        # Load data to get vocabulary
        _, _, self.vocab = self.data_loader.load_data()
        
        # Update model config with actual vocab size from data
        self.model_config.vocab_size = self.vocab.vocab_size
        
        # Initialize model with correct vocab_size
        self.model = GPT(
            vocab_size=self.model_config.vocab_size,
            n_embd=self.model_config.embed_dim,
            block_size=self.model_config.max_context,
            n_head=self.model_config.num_heads,
            n_layer=self.model_config.num_blocks,
            dropout=self.model_config.dropout
        )
        self.model = self.model.to(self.device)
    
    def load_model(self) -> None:
        """
        Load the trained model from checkpoint.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If model is not initialized (vocabulary not loaded).
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_vocabulary() first.")
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Please train the model first using train.py"
            )
        
        # Load state dictionary
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        # Load state dictionary into model
        self.model.load_state_dict(state_dict)
        
        # Set model to evaluation mode
        self.model.eval()
        print(f"Model loaded from {self.model_path}")
    
    def generate(
        self,
        max_new_tokens: int = 10000,
        start_tokens: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> str:
        """
        Generate text using the model.
        
        Args:
            max_new_tokens: Maximum number of tokens to generate.
            start_tokens: Optional starting tokens. If None, starts with zero token.
            temperature: Sampling temperature. Higher values make output more random.
        
        Returns:
            Generated text as a string.
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not loaded. Call load_vocabulary() first.")
        
        if self.model is None:
            raise ValueError("Model not initialized. Call load_vocabulary() first.")
        
        # Initialize with start tokens or zero
        if start_tokens is None:
            idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        else:
            idx = start_tokens.to(self.device).unsqueeze(0) if start_tokens.dim() == 1 else start_tokens.to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop idx to the last block_size tokens
                idx_cond = idx[:, -self.model_config.max_context:]
                
                # Get the predictions
                logits, _ = self.model(idx_cond)
                
                # Focus only on the last time step
                logits = logits[:, -1, :]  # (B, C)
                
                # Apply temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)  # (B, C)
                
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                
                # Append sampled index to the running sequence
                idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
        
        elapsed_time = time.time() - start_time
        print(f"Generated {max_new_tokens} tokens in {elapsed_time:.2f}s")
        
        # Decode the generated sequence
        generated_text = self.vocab.decode(idx[0].tolist())
        return generated_text
    
    def generate_to_file(
        self,
        output_path: str = "output.txt",
        max_new_tokens: int = 10000,
        start_tokens: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> None:
        """
        Generate text and save to file.
        
        Args:
            output_path: Path to save generated text.
            max_new_tokens: Maximum number of tokens to generate.
            start_tokens: Optional starting tokens.
            temperature: Sampling temperature.
        """
        generated_text = self.generate(max_new_tokens, start_tokens, temperature)
        
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(generated_text)
        
        print(f"Generated text saved to {output_path}")


def get_device() -> str:
    """
    Determine the best available device.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def main():
    """Main generation function."""
    # Initialize configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # Get device
    device = get_device()
    print(f"Device being used for generation: {device}")
    
    # Initialize generator
    generator = Generator(
        model_config=model_config,
        data_config=data_config,
        device=device
    )
    
    # Load vocabulary
    generator.load_vocabulary()
    
    # Load model
    generator.load_model()
    
    # Generate text
    generator.generate_to_file(max_new_tokens=10000)


if __name__ == "__main__":
    main()
