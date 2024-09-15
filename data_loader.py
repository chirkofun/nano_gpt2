import torch
import tiktoken
import numpy as np
import os

class DataLoaderLite:
  def __init__(self, B, T):
    self.B = B
    self.T = T

    # load tokens form disk and store them in memory
    with open('data/input.txt', 'r') as f:
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
      self.current_position = 0
    return x, y
  
  def get_batch(self, split, device):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    B, T = self.B, self.T
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - T, (B,))
    x = torch.stack([torch.from_numpy((data[i:i+T]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+T]).astype(np.int64)) for i in ix])
    if device.type == 'cuda' or device.type == 'mps':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y