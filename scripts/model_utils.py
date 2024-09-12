import torch
import os

# checkpoint save
def save_checkpoint(model, optimizer, step, loss, model_path="../trained_model"):
    checkpoint = {
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'step': step,
          'loss': loss.item(),
      }
    
    # Check if directory exists, create if it doesn't
    os.makedirs(model_path, exist_ok=True)
     
    torch.save(checkpoint, f"{model_path}/checkpoint_step_{step}.pth")
    print(f"Checkpoint saved at step {step}")


# write a function to load the model
def load_weights(model, model_path="../trained_model/gpt2_trained_weights.pth"):
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Failed to load weights: {e}. Starting with random weights.")
    else:
        print(f"No weights found at {model_path}. Starting with random weights.")
    return model

# write a function to save the model
def save_weights(model, model_path="../trained_model/gpt2_trained_weights.pth"):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def prepare_model(model, device):
    model.train()
    model = torch.compile(model, backend="aot_eager")
    model.to(device)
    print("Model compiled!")
    return model