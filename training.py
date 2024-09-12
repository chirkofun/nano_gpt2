import time
import inspect
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import os

import torch
import torchinfo

from model import GPT
from config.config import GPTConfig
from data_loader import DataLoaderLite
from utils import get_lr

if torch.cuda.is_available():
  device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
  device = torch.device("mps")
else:
  device = torch.device("cpu")

print(f"Using device: {device}")


from torchinfo import summary
model = GPT(GPTConfig()).to(device)
summary(model, input_size=(4, 64), dtypes=[torch.long])

#----------------------------------------------------------
# Set batch sizes and calculate gradient accumulation steps
#total_batch_size = 524288 # 2**19, around 0.5M in number of tokens
B = 32 # micro-batch size, best used is 64
T = 64 # sequence length
total_batch_size = B*T
assert total_batch_size % (B * T) == 0, "Total batch size should be divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
# Get DataLoader
train_loader = DataLoaderLite(B=B, T=T)
#----------------------------------------------------------
# Set precision for float and prepare the model
torch.set_float32_matmul_precision('high')
model.train()

# Load weights if exist
weights_path = "gpt2_model_weights.pth"
if os.path.exists(weights_path):
  try:
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"Loaded weights from {weights_path}")
  except Exception as e:
    print(f"Failed to load weights: {e}. Starting with random weights.")
else:
  print(f"No weights found at {weights_path}. Starting with random weights.")

model = torch.compile(model, backend="aot_eager")
model.to(device)
print("Model compiled!")
torch.manual_seed(42)
torch.mps.manual_seed(42)
#----------------------------------------------------------
# warmup + decay implementation
max_lr = 3e-3 # best used 3e-3
min_lr = 3e-4 # best used 3e-5
warmup_steps = 500 # best used 10
max_steps = 8000 # best used 2000
#----------------------------------------------------------
print("Start training:")
loss_history = []
num_steps = 8000
# Do the optimization
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
for step in range(num_steps):
  t0 = time.time()
  optimizer.zero_grad()
  loss_accum = 0
  for micro_step in range(grad_accum_steps):
      # get the next batch of data
      x, y = train_loader.next_batch()
      x, y = x.to(device), y.to(device)
      with torch.autocast(device_type='mps', dtype=torch.float16):
        logits, loss = model(x,y)
      loss = loss / grad_accum_steps
      loss_accum += loss.detach()
      loss.backward()

  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping

  # perform learning rate warmup and decay
  lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  optimizer.step()

  torch.mps.synchronize()
  t1 = time.time()
  dt = t1 - t0 # time diff in seconds

  tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
  tokens_per_sec = tokens_processed / dt

  loss_history.append(loss_accum)
  print(f"step {step} | loss: {loss_accum.item()} | lr: {lr} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/s: {tokens_per_sec:.2f}")


#----------------------------------------------------------
# Convert loss_history tensors to CPU and then to float
loss_history_cpu = [item.cpu().item() for item in loss_history]
# Plot loss curve
data = {
  'Epoch': list(range(0,len(loss_history_cpu))),
  'Loss': loss_history_cpu
}
df = pd.DataFrame(data)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Epoch', y='Loss')

# Customize the plot
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)