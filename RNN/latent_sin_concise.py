import torch
import random
from torch import optim, nn
import matplotlib.pyplot as plt

T = 100
batch_size = 16
train_T = 6000
tot_T = 10000
epoch_num = 20
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
    self.rnn = nn.RNN(1, hidden_num, 1, nonlinearity='relu')
    self.output = nn.Linear(hidden_num, 1)
    nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
    nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
    nn.init.xavier_uniform_(self.output.weight)

    self.rnn.bias_hh_l0.data.fill_(0)
    self.rnn.bias_ih_l0.data.fill_(0)
    self.output.bias.data.fill_(0)

  
  def __call__(self, X, state):
    Y, state = self.rnn(X.reshape((X.shape[0], X.shape[1], 1)), state)
    Y = self.output(Y)
    return Y, state
  
  def parameters(self):
    return list(self.rnn.parameters()) + list(self.output.parameters())
  
def grad_clipping(net, theta):
  params = net.parameters()
  norm = torch.sqrt(sum([torch.sum(p.grad ** 2) for p in params]))
  if norm > theta:
    for param in params:
      param.grad[:] *= theta / norm


net = RNN(256)
trainer = optim.Adam(net.parameters(), lr=0.001)
# trainer = optim.SGD(net.net, lr=0.004)
loss = nn.MSELoss()

for i in range(epoch_num):
  acc_loss = 0
  state = torch.zeros((1, batch_size, 256))
  for x, y in sequential_sampling(Y[:train_T], batch_size, T):
    trainer.zero_grad()
    x = x.permute((1, 0))
    y = y.T
    y = y.reshape((y.shape[0], y.shape[1], 1))
    y_hat, state = net(x, state)
    l = loss(y_hat, y)
    l.backward()
    state.detach_()
    grad_clipping(net, 1)
    trainer.step()

    acc_loss += l 
  print(f"epoch {i} acc_loss: {acc_loss}")

# net.clear_state(1)
state = torch.zeros((1, 1,256))
record = torch.zeros((tot_T, ))
for i, y in enumerate(Y[:train_T//2]):
  record[i], state = net(y.reshape((1, -1)), state)
for i in range(train_T//2, train_T):
  record[i], state = net(record[i-1].reshape((1,-1)), state)
for i in range(train_T, tot_T):
  record[i], state = net(record[i-1].reshape((1,-1)), state)
plt.plot(time, Y)
plt.plot(time, record.detach().numpy())
plt.show()