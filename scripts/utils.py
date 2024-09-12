import math

# calculate learning rate with warmup and cosine decay
def get_lr(it, warmup_steps, max_steps, max_lr, min_lr):
  # linear warmup
  if it < warmup_steps:
    return max_lr * (it+1) / warmup_steps
  if it > max_steps:
    return min_lr
  # cosine decay
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)