import torch 
import numpy as np

# create data 
epoch_num = 10
learning_rate = 0.3
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]])

# define network: (input_num, output_num)
structure = [(2, 2), (2, 1)]

layers = []
inputs = [] # input to each layer, used in backward propagation
for input, output in structure:
  layers.append((torch.rand(input, output), torch.zeros(output)))


# define forward, loss, train function
def relu(X):
  return torch.max(torch.zeros(X.shape), X)

def forward(X, layers):
  for W, b in layers[:-1]:
    inputs.append(X)
    X = torch.mm(X, W) + b
    X = relu(X)
  inputs.append(X)
  W, b = layers[-1]
  X = torch.mm(X, W) + b
  return X

def loss(y_hat, y):
  return sum((y_hat - y) ** 2) / (2 * 4)

# todo: did not update layers info
def train(layers, y_hat, y):
  def step(v):
    step_v = []
    for i, x in enumerate(v):
      step_v.append(1 if x >= 0 else 0)
    return torch.tensor(step_v)
  W2, b2 = layers[-1]
  W1, b1 = layers[0]
  x2 = inputs[-1]
  W2[:] -= torch.mm((y_hat - y).T, x2).reshape((-1, 1)) * learning_rate / 4
  b2[:] -= sum(y_hat - y) * learning_rate / 4
  x1 = inputs[0]
  z1 = torch.mm(x1, W1) + b1
  step_z1 = torch.tensor([ step(v).numpy() for v in z1])
  W1[:] -= torch.mm((y_hat - y).T, x1) * torch.mm(step_z1.float(), W2) * learning_rate / 4
  b1[:] -= sum(y_hat - y) * torch.mm(step_z1.float(), W2) * learning_rate / 4


# training
for i in range(epoch_num):
  acc_loss = 0
  y_hat = forward(X, layers)
  l = loss(y_hat, y)
  train(layers, y_hat, y)
  print(f"epoch {i+1} loss: {l}")
  acc_loss += l 

print("parameters: ")
for W, b in layers:
  print(f"W: {W}, b: {b}")

