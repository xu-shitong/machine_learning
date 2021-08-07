import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import torch.nn as nn
import torch.optim as optim

# define device used
if torch.cuda.is_available():
  dev = 'cuda:0'
else:
  dev = 'cpu'

device = torch.device(dev)

# define super parameters
sample_size = 1000
feature_num = 3
epoch_num = 15
batch_size = 10
learning_rate = 0.3

# define training data 
true_w = torch.tensor([4, 2.3, 6]).reshape((-1, 1)).to(device)
true_b = torch.tensor(0.5).to(device)
training_feature = torch.normal(0, 0.1, size=(sample_size, feature_num)).to(device)
training_label = torch.mm(training_feature, true_w) + true_b.to(device)
training_label += torch.normal(0, 0.01, size=(sample_size, 1)).to(device)
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

# define network 
net = nn.Sequential()
net.add_module('layer1', nn.Linear(feature_num, 1))

loss = nn.MSELoss()
trainer = optim.SGD(net.parameters(), lr=learning_rate)

# train
for i in range(epoch_num):
  trainer.zero_grad()
  acc_loss = 0
  for X, y in dataiter:
    y_hat = net(X)
    l = loss(y_hat, y)
    l.backward()
    trainer.step()
    acc_loss += l
  print(f"epoch {i+1} has acc loss: {acc_loss}")

print(f"final parameters: {net.parameters()}")
