import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas


batch_size = 3
epoch_num = 15
learning_rate = 0.3

# create dataset
# f(x) = 0 when x < 3
#      = 1 when x >= 3
training_features = torch.from_numpy(np.c_[range(-10, 10)])
training_labels = torch.from_numpy(np.c_[[0]*13 + [1]*7])
dataset = torch.utils.data.TensorDataset(training_features, training_labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# visualize data points
training_labels_array = training_labels.numpy().tolist()
training_features_array = training_features.numpy().tolist()
data = pandas.DataFrame(data={'feature': training_features_array, 'label': training_labels_array})
data.plot(kind='scatter', x='feature', y='label')
plt.show()

# initialize model 
W = torch.ones(1,1)
b = torch.zeros(1)

# define forward, loss, regression function
def forward(x, W, b):
  return 1 / (1 + (-x * W - b).exp())

def loss(p_hat, y):
  return - sum(y*p_hat.log() + (1-y)*(1-p_hat).log()) / batch_size

def train(W, b, p_hat, x, y):
  W[:] = W - sum((p_hat - y) * x) * learning_rate / batch_size
  b[:] = b - sum((p_hat - y)) * learning_rate / batch_size

# train
for i in range(epoch_num):
  acc_loss = 0
  for feature, label in dataloader:
    p_hat = forward(feature, W, b)
    l = loss(p_hat, label)
    train(W, b, p_hat, feature, label)
    acc_loss += l
  print(f"epoch {i} acc loss: {acc_loss}")

print(f"final parameter: W: {W}, b: {b}")
