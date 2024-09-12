from dataclasses import dataclass

@dataclass(frozen=True)
class GPTConfig:
  block_size: int = 64 # max sequence length
  vocab_size: int = 50304 # num of tokens
  n_embd: int = 128 # embedding dimension
  n_head: int = 4 # num of heads
  n_layer: int = 4 # num of layers