from matplotlib.collections import QuadMesh
import torch 
import matplotlib.pyplot as plt
from torch import nn, optim


T = 1000
tau = 4
epoch_num = 20
time = torch.arange(T) / 100
y = time.sin() + torch.normal(0, 0.1, time.shape)

# generate markov segments
feature = torch.zeros((T - tau, tau))
for i in range(tau):
    feature[:, i] = y[i: T - tau + i]

label = y[tau : ].reshape((-1, 1))

# create dataset
dataset = torch.utils.data.TensorDataset(feature, label)
dataiter = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


# create net 
net = nn.Sequential(
  nn.Linear(tau, 10), 
  nn.ReLU(),
  nn.Linear(10, 1)
)
net.apply(init_weights)
trainer = optim.Adam(net.parameters(), lr=0.01)
loss = nn.MSELoss(reduction='sum')

# train
for i in range(epoch_num):
  acc_loss = 0
  for X, Y in dataiter:
    y_hat = net(X)
    trainer.zero_grad()
    l = loss(y_hat, Y)
    l.backward()
    trainer.step()
    
    acc_loss += l
  print(f"epoch {i} acc_loss: {acc_loss}")


# predict using 4 given data and generate the rest
predicts = torch.zeros((T, ))
predicts[:4] = y[:4]
for i in range(4, T):
  predicts[i] = net(predicts[i-4 : i].reshape((-1, tau))).item()


# plot
plt.plot(time[tau:], label)
plt.plot(time[tau:], net(feature).detach().numpy())
plt.plot(time, predicts)
plt.show()