from numpy.core.fromnumeric import shape
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

# define hyper parameters
feature_num = 2
sample_num = 40
batch_size = 10

# define dataset
raw_data = torch.normal(mean=0, std=1, size=(batch_size, feature_num)) # each cluster take the same pattern of separation
cluster1 = raw_data + torch.tensor([2, 4] + ([1] * (feature_num - 2))) # first cluster of data centered at (2, 4)
cluster2 = raw_data + torch.tensor([5, 2] + ([1] * (feature_num - 2))) # second cluster data center at (5, 2)
cluster3 = raw_data + torch.tensor([4, 4] + ([1] * (feature_num - 2))) # third cluster centerred at (4, 4)
cluster4 = raw_data + torch.tensor([5, 4] + ([1] * (feature_num - 2))) # third cluster centerred at (5, 4)

feature_set = torch.cat([cluster1, cluster2, cluster3, cluster4])
label_set = torch.tensor([1] * (batch_size * 2) + [-1] * (batch_size * 2))

torch.save(feature_set, 'SVM/svm_features.log')
torch.save(label_set, 'SVM/svm_labels.log')

# change data to numpy data, easy for later calculation
feature_set = feature_set.type(torch.float64).numpy()
label_set = label_set.type(torch.float64).numpy().reshape((-1, 1))

# define dual problem parameters, calculate vector a
dot_result = np.matmul(feature_set, feature_set.T)
P = label_set.T * dot_result * label_set + np.diag([0.0001] * sample_num)

P = cvxopt_matrix(P)
q = cvxopt_matrix(-np.ones((sample_num, 1)))
G = cvxopt_matrix(-np.eye(sample_num))
h = cvxopt_matrix(np.zeros(sample_num))
A = cvxopt_matrix(label_set.T)
b = cvxopt_matrix(np.zeros(1))

sol = cvxopt_solvers.qp(P, q, G, h, A, b)
ans = np.array(sol['x'])
print(f"alpha = {ans}")
print(f"zero check = {np.dot(label_set.T, ans)}")

# filter only alphas with positive value
ind = (ans > 1e-4).flatten()
alpha = ans[ind]
sv = feature_set[ind]
sv_y = label_set[ind]

# parameters from unfilterred alpha
W_ = feature_set * label_set.reshape((-1, 1)) * ans.reshape((-1, 1))
W_ = W_.T.sum(axis=1)
b_ = label_set - (feature_set * W_).sum(axis=1)
b_ = b_.mean()
print(f"W_ = {W_}, b_ = {b_}")
print(f"classification on training samples: {np.dot(feature_set, W_) + b_}")

# parameters from filterred alpha
W = sv * sv_y.reshape((-1, 1)) * alpha.reshape((-1, 1))
W = W.T.sum(axis=1)
b = sv_y - (sv * W).sum(axis=1)
b = b.mean()
print(f"W = {W}, b = {b}")
print(f"classification on training samples: {np.dot(feature_set, W) + b}")


# printing seperating line on graph
plt.scatter(cluster1[:, 0].tolist(), cluster1[:, 1].tolist(), color='blue')
plt.scatter(cluster2[:, 0].tolist(), cluster2[:, 1].tolist(), color='blue')
plt.scatter(cluster3[:, 0].tolist(), cluster3[:, 1].tolist(), color='red')
plt.scatter(cluster4[:, 0].tolist(), cluster4[:, 1].tolist(), color='red')
x = np.linspace(0, 15, 100)
plt.plot(x, (-b - W[0]*x)/W[1], color='blue')
plt.plot(x, (-b_ - W_[0]*x)/W_[1], color='black')
plt.show()