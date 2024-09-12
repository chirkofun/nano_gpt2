import torch
import tiktoken

class DataLoaderLite:
  def __init__(self, B, T):
    self.B = B
    self.T = T

    # load tokens form disk and store them in memory
    with open('input.txt', 'r') as f:
      text = f.read()
    enc = tiktoken.get_encoding('gpt2')
    self.tokens = torch.tensor(enc.encode(text))
    print(f"Loaded {len(self.tokens)} tokens")
    print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

    # state
    self.current_position = 0

  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position:self.current_position + B * T + 1]
    x = buf[:-1].view(B,T) # inputs
    y = buf[1:].view(B,T) # targets
    # advance the position of the tensor
    self.current_position += B * T
    # if loading the next batch would be out of bounss, then reset
    if self.current_position + self.B * self.T >= len(self.tokens):
      print(self.current_position)
      self.current_position = 0
    return x, y