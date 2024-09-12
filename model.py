import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo
from torch._dynamo import debug_utils 

import math

# Self-Attention block
class CausalSelfAttention(nn.ModuleDict):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd & config.n_head == 0

    # key, query, value projections for all heads, but in a batch
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1
    # regularization
    self.n_head = config.n_head
    self.n_embd = config.n_embd

    # creating a mask for setting attention of future tokens to 0
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
    .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and mode head forward
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

    # attention (materializes the large (T, T) matrix for all queries and keys)
    att = (q @ k.transpose(-2,-1))*(1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    #y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    y = self.c_proj(y)
    return y

# Multi-layer Perceptron
class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # expansion layer
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # projection layer
    self.c_proj.NANOGPT_SCALE_INIT = 1

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
  
class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd), # word token embedding
        wpe = nn.Embedding(config.block_size, config.n_embd), # word position embedding
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden layers
        ln_f = nn.LayerNorm(config.n_embd)
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # weight sharing scheme
    self.transformer.wte.weight = self.lm_head.weight

    # init params
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      # scale down the weights on 1/sqrt(N) where N is number of residual layers
      if hasattr(module, 'NANOGPT_SCALE_INIT'):
        std *= (2 * self.config.n_layer) ** -0.5
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.size()

    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is"

    # forward the token and position embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
    pos_emb = self.transformer.wpe(pos)
    tok_emb = self.transformer.wte(idx)
    x = tok_emb + pos_emb

    # forward the blocks of the transformer
    for block in self.transformer.h:
      x = block(x)

    # forward the final layernorm and classifier
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)

    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
