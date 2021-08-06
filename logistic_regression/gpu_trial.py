import torch
from torch import nn

print(torch.cuda.is_available())

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)
X = torch.tensor([1,2]).reshape((1,2)).to(device)

print(device)
print(X)
print(X.is_cuda)

print(f"X get_device() is {X.get_device()}")

## try if cuda GPU times CPU tensor 
## runtime error: expect all tensor on same device
# Y = torch.arange(2).reshape((2, 1))
# Y = torch.mm(X, Y)

# try net on GPU
net = nn.Linear(2, 1).to(device)
out = net(X.reshape((1, -1)).float())
print(out)

# try dataiter on GPU
dataset = torch.utils.data.TensorDataset(torch.tensor([1]*10, device=device), torch.tensor([2]*10, device=device))
dataiter = torch.utils.data.DataLoader(dataset, batch_size=2)
for X, y in dataiter:
    break
print(X)
print(net(X.reshape((1, -1)).float()))

