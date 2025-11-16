"""Training script for GPT model."""

import time
import torch
from typing import Dict, Optional

from model import GPT
from data import DataLoader
from config import ModelConfig, TrainingConfig, DataConfig


class Trainer:
    """Handles training of the GPT model."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        device: str,
        model_save_path: str = "GPT_model.pt"
    ):
        """
        Initialize the Trainer.
        
        Args:
            model_config: ModelConfig instance with model hyperparameters.
            training_config: TrainingConfig instance with training hyperparameters.
            data_config: DataConfig instance with data settings.
            device: Device to train on ('cuda', 'mps', or 'cpu').
            model_save_path: Path to save the trained model.
        """
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.device = device
        self.model_save_path = model_save_path
        
        # Set random seed
        torch.manual_seed(training_config.seed)
        
        # Initialize data loader
        self.data_loader = DataLoader(data_config)
        
        # Initialize model
        self.model = GPT(
            vocab_size=model_config.vocab_size,
            n_embd=model_config.embed_dim,
            block_size=model_config.max_context,
            n_head=model_config.num_heads,
            n_layer=model_config.num_blocks,
            dropout=model_config.dropout
        )
        self.model = self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # Training state
        self.train_data = None
        self.val_data = None
        self.vocab = None
    
    def prepare_data(self) -> None:
        """Download and load training data."""
        # Download data if needed
        data_path = self.data_loader.download_data()
        if not data_path:
            raise RuntimeError("Data download was unsuccessful")
        
        # Load data
        self.train_data, self.val_data, self.vocab = self.data_loader.load_data()
        print(f"Data loaded: {len(self.train_data)} train samples, {len(self.val_data)} val samples")
    
    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """
        Estimate loss on train and validation sets.
        
        Returns:
            Dictionary with 'train' and 'val' loss values.
        """
        out = {}
        self.model.eval()
        
        for split in ['train', 'val']:
            losses = torch.zeros(self.training_config.eval_iters)
            for k in range(self.training_config.eval_iters):
                x, y = self.data_loader.sample_batch(
                    split=split,
                    max_context=self.model_config.max_context,
                    batch_size=self.training_config.batch_size
                )
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        
        self.model.train()
        return out
    
    def train(self) -> None:
        """Train the model."""
        print("Starting training")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
        
        start_time = time.time()
        
        for iter in range(self.training_config.num_epochs):
            # Evaluate loss periodically
            if iter % self.training_config.validation_freq == 0:
                losses = self.estimate_loss()
                elapsed_time = time.time() - start_time
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}, "
                    f"time: {elapsed_time:.2f}s"
                )
            
            # Sample a batch of data
            xb, yb = self.data_loader.sample_batch(
                split='train',
                max_context=self.model_config.max_context,
                batch_size=self.training_config.batch_size
            )
            xb, yb = xb.to(self.device), yb.to(self.device)
            
            # Forward pass
            logits, loss = self.model(xb, yb)
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
    
    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save the trained model.
        
        Args:
            path: Optional path to save the model. If None, uses model_save_path.
        """
        save_path = path or self.model_save_path
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


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
    """Main training function."""
    # Initialize configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # Get device
    device = get_device()
    print(f"Device being used to train: {device}")
    
    # Initialize trainer
    trainer = Trainer(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        device=device
    )
    
    # Prepare data
    trainer.prepare_data()
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model()


if __name__ == "__main__":
    main()
