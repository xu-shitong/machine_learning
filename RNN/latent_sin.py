import torch
import random
from torch import optim, nn
import matplotlib.pyplot as plt

T = 100
batch_size = 16
train_T = 6000
tot_T = 10000
epoch_num = 50
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

def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # yield X, Y
        yield torch.vstack(X), torch.vstack(Y)
# net
class RNN:
  def __init__(self, hidden_num) -> None:
    self.hidden_num = hidden_num
    self.net = [torch.zeros((1, hidden_num)), torch.zeros((hidden_num, hidden_num)), torch.zeros((hidden_num)), torch.zeros((hidden_num, 1)), torch.zeros((1))]
    nn.init.xavier_uniform_(self.net[0])
    nn.init.xavier_uniform_(self.net[1])
    nn.init.xavier_uniform_(self.net[3])
    for param in self.net:
      param.requires_grad_(True)
  
  def __call__(self, X, state):
    print(X.shape)
    W_xh, W_hh, b_h, W_hq, b_q = self.net
    Y = torch.zeros(X.shape)
    for i, x in enumerate(X):
      x = x.reshape((-1, 1))
      # ReLU generalize the best on function with periodicity, (e.g. sin), than Sigmoid, Tanh
      state = nn.ReLU()(torch.mm(x, W_xh) + torch.mm(state, W_hh) + b_h)
      Y[i, :] = (torch.mm(state, W_hq) + b_q).T
    return Y, (state, )
  
def grad_clipping(net, theta):
  params = net.net
  norm = torch.sqrt(sum([torch.sum(p.grad ** 2) for p in params]))
  if norm > theta:
    for param in params:
      param.grad[:] *= theta / norm


net = RNN(256)
trainer = optim.Adam(net.net, lr=0.001)
# trainer = optim.SGD(net.net, lr=0.004)
loss = nn.MSELoss()

for i in range(epoch_num):
  acc_loss = 0
  state = torch.zeros((batch_size, 256))
  for x, y in seq_data_iter_random(Y[:train_T], batch_size, T):
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