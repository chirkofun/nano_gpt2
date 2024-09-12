import torch
import time
from tqdm import tqdm
from configs.config import GPTConfig, TrainingConfig
from model import GPT
from data_loader import DataLoaderLite
from scripts.utils import get_lr, setup_device
from scripts.model_utils import load_model_weights, save_checkpoint, save_model_weights, plot_training_results
from scripts.train import train_step, evaluate_model

def main():
    model_config = GPTConfig()
    training_config = TrainingConfig()
    device = setup_device()
    model = setup_model(model_config, device)
    train_loader = DataLoaderLite(B=training_config.batch_size, T=training_config.block_size)
    optimizer = setup_optimizer(model, training_config)
    
    training_results = train_loop(model, train_loader, optimizer, training_config, model_config, device)
    
    save_model_weights(model, path=model_config.weights_path)
    plot_training_results(training_results, path=model_config.plots_path)

def setup_model(model_config, device):
    model = GPT(model_config).to(device)
    model = load_model_weights(model, path=model_config.weights_path)
    model = torch.compile(model, backend="aot_eager")
    return model

def setup_optimizer(model, training_config):
    return torch.optim.AdamW(model.parameters(), **training_config.optimizer_params)

def print_training_stats(step, step_result):
    print(f"Step {step}: "
          f"loss = {step_result['loss']:.4f} | "
          f"lr = {step_result['lr']:.6f} | "
          f"norm = {step_result['norm']:.4f} | "
          f"tokens/sec = {step_result['tokens_per_sec']:.2f}")

def train_loop(model, train_loader, optimizer, training_config, model_config, device):
    losses = []
    norms = []

    print("Starting training...")
    start_time = time.time()
    for step in tqdm(range(training_config.num_steps), desc="Training"):
        step_result = train_step(model, train_loader, optimizer, training_config, device, step)
        
        losses.append(step_result['loss'])
        norms.append(step_result['norm'])

        if step % training_config.print_interval == 0:
            print_training_stats(step, step_result)

        if step % training_config.save_interval == 0 and step != 0:
            save_checkpoint(model, optimizer, step, step_result['loss'], path = model_config.checkpoints_path)

    training_results = {
        'Loss': losses,
        'Norm': norms,
    }
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    return training_results

if __name__ == "__main__":
    main()