
import torch

print("PyTorch Version:", torch.__version__)
print("Is MPS (Apple GPU) available?:", torch.backends.mps.is_available())

# Test it by moving a calculation to your Mac's GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(3, 3, device=device)
    print("Success! Tensor moved to Mac GPU:", x.device)
else:
    print("Running on CPU only.")

x = torch.empty(3, 2, 3, 4)
print(x)
print(x.size())