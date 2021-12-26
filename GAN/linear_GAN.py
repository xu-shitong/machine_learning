import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
import torch.optim as optim
import matplotlib.pyplot as plt

# hyperparameter
epoch = 100

# dataset
raw_x = torch.normal(0.0, 1, (1000, 2), dtype=torch.float32)
A = torch.tensor([[1, 2], [-0.1, 0.5]], dtype=torch.float32)
b = torch.tensor([1, 2])
X = torch.matmul(raw_x, A) + b

# # visualization
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()


dataset = torch.utils.data.TensorDataset(X, torch.ones((X.shape[0], 1), dtype=torch.float32))
dataiter = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

# D = nn.Sequential(
#   nn.Linear(in_features=2, out_features=1),
#   nn.Sigmoid()
# )
D = nn.Sequential(
    nn.Linear(2, 5), nn.Tanh(),
    nn.Linear(5, 3), nn.Tanh(),
    nn.Linear(3, 1))
trainer_d = optim.Adam(D.parameters(), lr=0.05)

G = nn.Sequential(
  nn.Linear(in_features=2, out_features=2)
)
trainer_g = optim.Adam(G.parameters(), lr=0.005)
# loss = nn.CrossEntropyLoss()
loss = nn.BCEWithLogitsLoss(reduction='sum')

for i in range(epoch):
  acc_d_loss = 0
  acc_g_loss = 0
  for X, y in dataiter:
    X_prime = torch.normal(0, 1, size=X.shape, dtype=torch.float32)

    # D part
    trainer_d.zero_grad()
    z = G(X_prime)
    y_prime_hat = D(z.detach())
    y_real_hat = D(X)
    l_d = (loss(y_real_hat, torch.zeros(y.shape)) + loss(y_prime_hat, torch.zeros(y.shape, dtype=torch.float32))) / 2
    l_d.backward()
    trainer_d.step()

    # G part
    trainer_g.zero_grad()
    y_prime_hat = D(z)
    l_g = loss(y_prime_hat, torch.ones(y.shape, dtype=torch.float32))
    l_g.backward()
    trainer_g.step()

    acc_d_loss += l_d
    acc_g_loss += l_g

  print(f"epoch {i} acc_d_loss: {acc_d_loss}, acc_g_loss: {acc_g_loss}")

print(f"final weight: g: w: \n{G[0].weight.data}, b: \n{G[0].bias.data}")
print(f"final weight: d: w: \n{D[0].weight.data}, b: \n{D[0].bias.data}")