"""Transformer-based language model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Head(nn.Module):
    """Single head of self-attention mechanism."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        """
        Initialize a single attention head.

        Args:
            n_embd: Embedding dimension.
            head_size: Size of each attention head.
            block_size: Maximum sequence length for causal masking.
            dropout: Dropout probability.
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention head.

        Args:
            x: Input tensor of shape (B, T, C) where B is batch size,
               T is sequence length, and C is embedding dimension.

        Returns:
            Output tensor of shape (B, T, head_size).
        """
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        
        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads: int, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        """
        Initialize multi-head attention.

        Args:
            num_heads: Number of attention heads.
            n_embd: Embedding dimension.
            head_size: Size of each attention head.
            block_size: Maximum sequence length for causal masking.
            dropout: Dropout probability.
        """
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head attention.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple feedforward linear layer followed by a non-linearity."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        """
        Initialize feedforward network.

        Args:
            n_embd: Embedding dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # From "Attention is All You Need" paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # Projection layer going back into residual pathway
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feedforward network.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd: int, n_head: int, head_size: int, block_size: int, dropout: float) -> None:
        """
        Initialize transformer block.

        Args:
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            head_size: Size of each attention head.
            block_size: Maximum sequence length for causal masking.
            dropout: Dropout probability.
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, n_embd, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block with residual connections.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        # Pre-norm architecture: normalize before attention/feedforward
        # Residual connections allow gradients to flow more easily through the network
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """Transformer-based bigram language model. Minimal GPT implementation"""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        n_head: int,
        n_layer: int,
        dropout: float
    ) -> None:
        """
        Initialize the language model.

        Args:
            vocab_size: Size of the vocabulary.
            n_embd: Embedding dimension.
            block_size: Maximum sequence length (context window).
            n_head: Number of attention heads.
            n_layer: Number of transformer blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        head_size = n_embd // n_head
        
        # Store hyperparameters
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout

        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, head_size, block_size, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and language model head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            idx: Input token indices of shape (B, T).
            targets: Target token indices of shape (B, T) for loss computation.
                    If None, loss will not be computed.

        Returns:
            Tuple of (logits, loss) where:
            - logits: Predicted logits of shape (B, T, vocab_size)
            - loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape

        # Token and position embeddings
        token_embd = self.token_embedding_table(idx)  # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = token_embd + pos_embd  # (B, T, C)
        
        # Apply transformer blocks
        x = self.blocks(x)  # (B, T, C)
        
        # Final layer norm and language model head
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss