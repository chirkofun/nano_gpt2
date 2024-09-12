import torch
import os
import matplotlib.pyplot as plt

# checkpoint save
def save_checkpoint(model, optimizer, step, loss, path="trained_model/checkpoints"):
    checkpoint = {
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'step': step,
          'loss': loss
      }
    
    # Check if directory exists, create if it doesn't
    os.makedirs(path, exist_ok=True)

    # Construct full file path
    file_path = os.path.join(path, f"checkpoint_step_{step}.pth")
     
    try:
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved at step {step} to {file_path}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")


# write a function to load the model
def load_model_weights(model, path="trained_model/gpt2_trained_weights.pth"):
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, weights_only=True))
            print(f"Loaded weights from {path}")
        except Exception as e:
            print(f"Failed to load weights: {e}. Starting with random weights.")
    else:
        print(f"No weights found at {path}. Starting with random weights.")
    return model

# write a function to save the model
def save_model_weights(model, path="trained_model/gpt2_trained_weights.pth"):
    torch.save(model._orig_mod.state_dict(), path)
    print(f"Model saved to {path}")

def prepare_model(model, device):
    model = torch.compile(model, backend="aot_eager")
    model.to(device)
    print("Model compiled!")
    return model

def plot_training_results(training_results, path="trained_model/plots"):
    os.makedirs(path, exist_ok=True)
    for key, values in training_results.items():
        values_cpu = [v.cpu().numpy() if torch.is_tensor(v) else v for v in values]
        steps = [i for i in range(len(values_cpu))]
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values_cpu, label=f'{key} (training)')
        plt.xlabel('Steps')
        plt.ylabel(key)
        plt.title(f'{key} Over Steps')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{path}/{key}.png")
        plt.close()