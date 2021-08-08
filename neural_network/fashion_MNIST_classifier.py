from sklearn.datasets import fetch_openml
import torch
from torch import nn
import torch.optim as optim
# import matplotlib.pyplot as plt

# define device used
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)


# define training superparameter
train_test_ratio = 0.9
leanring_rate = 0.3
batch_size = 10
epoch_num = 3

# get dataset from MNIST
mnist = fetch_openml('mnist_784', version=1)
feature_set = mnist['data']
label_set = mnist['target']
train_test_bound = int(len(feature_set) * train_test_ratio)
train_feature_set, train_label_set = feature_set[:train_test_bound], label_set[:train_test_bound]
test_feature_set, test_label_set = feature_set[train_test_bound:], label_set[train_test_bound:]

train_feature_set, test_feature_set = torch.tensor(train_feature_set.values, device=device), torch.tensor(test_feature_set.values, device=device)
train_label_set, test_label_set = torch.tensor(train_label_set.values.codes, device=device), torch.tensor(test_label_set.values.codes, device=device)

train_set = torch.utils.data.TensorDataset(train_feature_set, train_label_set)
train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# # visualise the first image
# plt.imshow(train_feature_set.iloc[0].values.reshape((28,28)))
# plt.show()

# define network, using LeNet
net = nn.Sequential(
  nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1), # 24*24
  nn.Sigmoid(),
  nn.MaxPool2d(kernel_size=2,stride=2), # 12*12
  nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # 8*8
  nn.Sigmoid(),
  nn.MaxPool2d(kernel_size=2, stride=2), # 4*4
  # flatten?
  nn.Flatten(),
  nn.Linear(4*4*16, 120),
  nn.Sigmoid(),
  nn.Linear(120, 84),
  nn.Sigmoid(),
  nn.Linear(84, 10),
  nn.Softmax()
)
net.to(device)

# define loss, trainer
loss = nn.CrossEntropyLoss()
trainer = optim.SGD(net.parameters(), lr=leanring_rate)

# train 
for i in range(epoch_num):

  acc_loss = 0
  for X, y in train_iter:
    y_hat = net(X.reshape((10, 1, 28, 28)).float())
    l = loss(y_hat, y.long())
    l.backward()
    acc_loss += l
    trainer.step()
  print(f"epoch {i+1} has loss {acc_loss}")

torch.save(net, 'fashion_MNIST_classifier.log')
print(f"write net data to fashion_MNIST_classifier.log")