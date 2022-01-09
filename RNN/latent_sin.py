import torch
import random
from torch import optim, nn
import matplotlib.pyplot as plt

T = 100
batch_size = 16
train_T = 6000
tot_T = 10000
epoch_num = 100
time = torch.arange(tot_T) / 20
# Y = time.sin() + torch.normal(0, 0.2, time.shape)
Y = time.sin() + 1

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
    # self.state = torch.zeros((batch_size, hidden_num))
    # self.net = [torch.normal(0, 1, (1, hidden_num)), torch.normal(0, 1, (hidden_num, hidden_num)), torch.zeros((hidden_num)), torch.normal(0, 1, (hidden_num, 1)), torch.zeros((1))]
    self.net = [torch.zeros((1, hidden_num)), torch.zeros((hidden_num, hidden_num)), torch.zeros((hidden_num)), torch.zeros((hidden_num, 1)), torch.zeros((1))]
    nn.init.xavier_uniform_(self.net[0])
    nn.init.xavier_uniform_(self.net[1])
    nn.init.xavier_uniform_(self.net[3])
    for param in self.net:
      param.requires_grad_(True)
  
  def __call__(self, X, state):
    W_xh, W_hh, b_h, W_hq, b_q = self.net
    Y = torch.zeros(X.shape)
    # state = self.state
    for i, x in enumerate(X):
      x = x.reshape((-1, 1))
      # ReLU generalize the best on function with periodicity, (e.g. sin), than Sigmoid, Tanh
      state = nn.ReLU()(torch.mm(x, W_xh) + torch.mm(state, W_hh) + b_h)
      Y[i, :] = (torch.mm(state, W_hq) + b_q).T
    # self.state = state.detach()
    return Y, (state, )
  
  # def clear_state(self, batch_size):
  #   self.state = torch.zeros((batch_size, self.hidden_num))

def grad_clipping(net, theta):
  params = net.net
  norm = torch.sqrt(sum([torch.sum(p.grad ** 2) for p in params]))
  if norm > theta:
    for param in params:
      param.grad[:] *= theta / norm


net = RNN(256, batch_size)
trainer = optim.Adam(net.net, lr=0.001)
# trainer = optim.SGD(net.net, lr=0.004)
loss = nn.MSELoss()

# buggy code: should not put state initialization here, 
#             however, this weirdly result in better generalization in sin value range prediction
state = torch.zeros((batch_size, 256))
for i in range(epoch_num):
  acc_loss = 0
  # # in contrast, should put state initialization here
  # state = torch.zeros((batch_size, 256))
  for x, y in sequential_sampling(Y[:train_T], batch_size, T):
    trainer.zero_grad()
    x = x.permute((1, 0))
    y = y.permute((1, 0))
    y_hat, (state, ) = net(x, state)
    l = loss(y_hat, y)
    l.backward()
    state.detach_()
    grad_clipping(net, 1)
    trainer.step()

    acc_loss += l 
  print(f"epoch {i} acc_loss: {acc_loss}")

# net.clear_state(1)
state = torch.zeros((1, 256))
record = torch.zeros((tot_T, ))
for i, y in enumerate(Y[:train_T//2]):
  record[i], (state, ) = net(y.reshape((1, -1)), state)
for i in range(train_T//2, train_T):
  record[i], (state, ) = net(record[i-1].reshape((1,-1)), state)
for i in range(train_T, tot_T):
  record[i], (state, ) = net(record[i-1].reshape((1,-1)), state)
plt.plot(time, Y)
plt.plot(time, record.detach().numpy())
plt.show()