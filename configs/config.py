from dataclasses import dataclass, field

@dataclass(frozen=True)
class GPTConfig:
  # model parameters
  block_size: int = 64 # max sequence length
  vocab_size: int = 50304 # num of tokens
  n_embd: int = 128 # embedding dimension
  n_head: int = 4 # num of heads
  n_layer: int = 4 # num of layers

  # model paths
  weights_path: str = "trained_model/gpt2_trained_weights.pth"
  checkpoints_path: str = "trained_model/checkpoints"
  plots_path: str = "trained_model/plots"


@dataclass
class TrainingConfig:
  # training parameters
  batch_size: int = 32 # batch size
  block_size: int = 64 # block size
  grad_accumulation_steps: int = 1 # gradient accumulation steps

  num_steps: int = 40000 # number of steps
  eval_interval: int = 100 # evaluate interval
  eval_iters: int = 100 # evaluate iterations

  max_lr: float = 3e-3 # max learning rate
  min_lr: float = 3e-4 # min learning rate
  warmup_steps: int = 500 # warmup steps
  max_steps: int = 40000 # max number of iterations
  optimizer_params: dict = field(default_factory=lambda: {
    'lr': 3e-4,
    'betas': (0.9, 0.95),
    'eps': 1e-8,
    'weight_decay': 0.1
  })

  print_interval: int = 100 # print every steps
  save_interval: int = 1000 # save interval