import torch
import time
from tqdm import tqdm
from configs.config import GPTConfig
from model import GPT
from data_loader import DataLoaderLite
from scripts.utils import get_lr, setup_device

def train_step(model, train_loader, optimizer, training_config, device, step):
    start_time = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for _ in range(training_config.grad_accumulation_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits, loss = model(x, y)
        loss = loss / training_config.grad_accumulation_steps
        loss_accum += loss.detach()
        loss.backward()

    # Clip gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Calculate learning rate
    lr = get_lr(step, training_config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Calculate tokens per second
    torch.mps.synchronize()
    processed_tokens = training_config.batch_size * training_config.block_size * training_config.grad_accumulation_steps
    tokens_per_sec = processed_tokens / (time.time() - start_time)
    step_result = {
        'loss': loss.item(),
        'lr': lr,
        'norm': norm,
        'tokens_per_sec': tokens_per_sec
    }

    return step_result

# TODO: refactor and use validation set
def evaluate_model(model, train_loader, config, device):
    model.eval()
    losses = []
    for _ in range(config.eval_iters):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits, loss = model(x, y)
        losses.append(loss.item())
    return torch.tensor(losses).mean().item()