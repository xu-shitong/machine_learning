import torch
import random
from torch import optim, nn
import matplotlib.pyplot as plt

T = 35
batch_size = 16
train_T = 600
epoch_num = 5
time = torch.arange(1000) / 100
Y = time.sin() + torch.normal(0, 0.2, time.shape)

# sequential sampling
def sequential_sampling(dataset, batch_size, step_size):
  start = random.randint(0, step_size)
  num_tokens = ((len(dataset) - start - 1) // step_size // batch_size) * step_size * batch_size
  Xs = dataset[start : start + num_tokens].reshape((batch_size, -1))
  Ys = dataset[start + 1 : start + num_tokens + 1].reshape((batch_size, -1))
  for i in range(0, step_size * (Xs.shape[1] // step_size), step_size):
    X = Xs[:, i: i + step_size]
    Y = Ys[:, i: i + step_size]
    yield X, Y

# net
class RNN:
  def __init__(self, hidden_num, batch_size) -> None:
    self.hidden_num = hidden_num
    self.state = torch.zeros((batch_size, hidden_num))
    self.net = [torch.normal(0, 1, (1, hidden_num)), torch.normal(0, 1, (hidden_num, hidden_num)), torch.zeros((hidden_num)), torch.normal(0, 1, (hidden_num, 1)), torch.zeros((1))]
    for param in self.net:
      param.requires_grad_(True)
    
  def __call__(self, x):
    W_xh, W_hh, b_h, W_hq, b_q = self.net
    for i in 
    h = torch.mm(x, W_xh) + torch.mm(self.state, W_hh) + b_h
    torch.mm(h, W_hq) + b_q, (h, )

  def clear_state(self, batch_size):
    self.state = torch.zeros((batch_size, self.hidden_num))

net = RNN(10, batch_size)
trainer = optim.Adam(net.parameters(), lr=0.01)
loss = nn.MSELoss()

for i in range(epoch_num):
  acc_loss = 0
  net.clear_state()
  for x, y in sequential_sampling(Y[:train_T], batch_size, T):
    trainer.zero_grad()
    y_hat, (state, ) = net(x)
    l = loss(y_hat, y)
    l.backward()
    trainer.step()

    acc_loss += l 
  print(f"epoch {i} acc_loss: {acc_loss}")

record = torch.zeros((T, ))
state = torch.zeros((1, 10))
for i, y in enumerate(Y[:train_T]):
  record[i], (state, ) = forward(y, state, net)
for i in range(T - train_T):
  record[i], (state, ) = forward(record)
plt.plot(time, Y)
plt.plot(time[:train_T], )
plt.show()