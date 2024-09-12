from math import cos, pi

# calculate learning rate with warmup and cosine decay
def get_lr(it, training_config):
  # linear warmup
  if it < training_config.warmup_steps:
    return training_config.max_lr * (it + 1) / training_config.warmup_steps
  if it > training_config.max_steps:
    return training_config.min_lr
  # cosine decay
  decay_ratio = (it - training_config.warmup_steps) / (training_config.max_steps - training_config.warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1 + cos(pi * decay_ratio))
  return training_config.min_lr + coeff * (training_config.max_lr - training_config.min_lr)

def setup_device():
  import torch
  if torch.cuda.is_available():
    device = torch.device("cuda")
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
  else:
    device = torch.device("cpu")
  print(f"Using device: {device}")
  return device