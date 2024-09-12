import time
import inspect
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import os

import torch
import torchinfo

from model import GPT
from config.config import GPTConfig
from data_loader import DataLoaderLite
from scripts.utils import get_lr
from scripts.model_utils import save_checkpoint, load_weights, save_weights, prepare_model

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

# Load weights if exist and prepare the model
weights_path = "gpt2_model_weights.pth"
model = load_weights(model, weights_path)
model = prepare_model(model, device)

torch.manual_seed(42)
torch.mps.manual_seed(42)
#----------------------------------------------------------
# warmup + decay implementation
max_lr = 3e-3 # best used 3e-3
min_lr = 3e-4 # best used 3e-5
warmup_steps = 500 # best used 10
max_steps = 8000 # best used 2000
num_steps = 8000
#----------------------------------------------------------
print("Start training:")
loss_history = []
learning_rate_history = []
norm_history = []
tokens_per_sec_history = []

start_time = time.time()
# Do the optimization
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
for step in tqdm(range(num_steps), desc="Training"):
  t0 = time.time()
  optimizer.zero_grad()
  loss_accum = 0
  for micro_step in range(grad_accum_steps):
      # get the next batch of data
      x, y = train_loader.next_batch()
      x, y = x.to(device), y.to(device)
      with torch.autocast(device_type=device.type, dtype=torch.float16):
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
  learning_rate_history.append(lr)
  norm_history.append(norm)
  tokens_per_sec_history.append(tokens_per_sec)

  if step % 1000 == 0 and step != 0:
    print(f"step {step} | loss: {loss_accum.item()} | lr: {lr} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/s: {tokens_per_sec:.2f}")
    save_checkpoint(model, optimizer, step, loss_accum)

end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")


#----------------------------------------------------------
# Convert loss_history tensors to CPU and then to float
loss_history_cpu = [item.cpu().item() for item in loss_history]
learning_rate_history_cpu = learning_rate_history
norm_history_cpu = [item.cpu().item() for item in norm_history]
tokens_per_sec_history_cpu = tokens_per_sec_history

# Create a DataFrame
data = {
    'Epoch': list(range(len(loss_history_cpu))),
    'Loss': loss_history_cpu,
    'Learning Rate': learning_rate_history_cpu,
    'Norm': norm_history_cpu,
    'Tokens per Second': tokens_per_sec_history_cpu
}
df = pd.DataFrame(data)

# Create and customize multiple plots
plt.figure(1, figsize=(10, 6))
sns.lineplot(data=df, x='Epoch', y='Loss')
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('trained_model/loss_plot.png')

plt.figure(2, figsize=(10, 6))
sns.lineplot(data=df, x='Epoch', y='Learning Rate')
plt.title('Learning Rate Over Time')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.savefig('trained_model/lr_plot.png')

plt.figure(3, figsize=(10, 6))
sns.lineplot(data=df, x='Epoch', y='Norm')
plt.title('Gradient Norm Over Time')
plt.xlabel('Epoch')
plt.ylabel('Norm')
plt.grid(True)
plt.savefig('trained_model/norm_plot.png')

plt.figure(4, figsize=(10, 6))
sns.lineplot(data=df, x='Epoch', y='Tokens per Second')
plt.title('Tokens per Second Over Time')
plt.xlabel('Epoch')
plt.ylabel('Tokens per Second')
plt.grid(True)
plt.savefig('trained_model/tokens_per_sec_plot.png')

print("Plots saved in trained_model directory")
#----------------------------------------------------------
save_weights(model)