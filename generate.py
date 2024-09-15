import tiktoken
import torch
import torch.nn.functional as F

from scripts.model_utils import load_model_weights
from model import GPT
from configs.config import GPTConfig

torch.mps.empty_cache()
torch.cuda.empty_cache()

num_return_sequences = 5
max_length = 64
device = 'mps'

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I am a language model")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

model = GPT(GPTConfig())
model = load_model_weights(model)

#def generate(model, num_return_sequences, max_length, device='cpu'):
model.eval()
model.to(device)
#torch.manual_seed(42)
#torch.mps.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x)
        logits = logits[:, -1, :] # last position
        probs = F.softmax(logits, dim=-1) # get probabilities
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # do top-k sampling of 50
        ix = torch.multinomial(topk_probs, 1) # select a token from top-k probabilities
        xcol = torch.gather(topk_indices, -1, ix) # gather the corresponding indices
        x = torch.cat((x, xcol), dim=1) # append to the sequence

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

#generate(model, num_return_sequences, max_length)