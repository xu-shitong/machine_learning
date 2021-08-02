import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)
X = torch.zeros(4,3).to(device)

print(device)
print(X)
print(X.is_cuda)