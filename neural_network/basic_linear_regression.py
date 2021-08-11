import torch
import numpy as np
# import pandas
# import matplotlib.pyplot as plt
# from pandas.plotting import scatter_matrix
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('../machine_learning/')
from utils.functions import train

# define device used
if torch.cuda.is_available():
  dev = 'cuda:0'
else:
  dev = 'cpu'

device = torch.device(dev)

# define super parameters
sample_size = 1000
feature_num = 3
epoch_num = 10
batch_size = 10
learning_rate = 0.3

# define training data 
true_w = torch.tensor([4, 2.3, 6], device=device).reshape((-1, 1))
true_b = torch.tensor(0.5, device=device)
training_feature = torch.normal(0, 0.1, size=(sample_size, feature_num), device=device)
training_label = torch.mm(training_feature, true_w) + true_b
training_label += torch.normal(0, 0.01, size=(sample_size, 1), device=device)
dataset = torch.utils.data.TensorDataset(training_feature, training_label)
dataiter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # plot sample
# training_feature0_array = training_feature.T[0].numpy().tolist()
# training_feature1_array = training_feature.T[1].numpy().tolist()
# training_feature2_array = training_feature.T[2].numpy().tolist()
# training_label_array = training_label.T[0].numpy().tolist()
# data = pandas.DataFrame(data={'feature0': training_feature0_array, 'feature1': training_feature1_array, 'feature2': training_feature2_array, 'label': training_label_array})
# # data.plot(kind='scatter', x='feature0', y='label')
# scatter_matrix(data, figsize=(12,10))
# plt.show()

# # define network method 1
# net = nn.Sequential()
# net.add_module('layer1', nn.Linear(feature_num, 1).to(device))

# define network method 2
W = torch.ones(feature_num,1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
class MyNet(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.weight = W
    self.bias = b

  def forward(self, X):
    return torch.mm(X, self.weight) + self.bias

net = MyNet()

loss = nn.MSELoss()
trainer = optim.SGD([W, b], lr=0.7, momentum=0.9)
# trainer = optim.Adam(net.parameters(), lr=0.7)

# train
# print(f"training using {dev}")
# for i in range(epoch_num):
#   acc_loss = 0
#   for X, y in dataiter:
#     y_hat = net(X)
#     l = loss(y_hat, y)
#     trainer.zero_grad()
#     l.backward()
#     trainer.step()
#     acc_loss += l
#     if i == 0: 
#       print(f"epoch 1 loss+ {l}")
#   print(f"epoch {i+1} has acc loss: {acc_loss}")

# print(f"final parameters: {net.parameters()}")
train(net, dataiter, loss, trainer, epoch_num)
print(f"parameters: W: {W}, \nb: {b}")

torch.save(net, 'parameter_log/basic_linear_regression.log')
print("data saved in parameter_log/basic_linear_regression.log")