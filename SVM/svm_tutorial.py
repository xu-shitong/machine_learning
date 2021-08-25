from numpy.core.fromnumeric import shape
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

# cvxopt_solvers.options['show_progress'] = False
# cvxopt_solvers.options['abstol'] = 1e-10
# cvxopt_solvers.options['reltol'] = 1e-10
# cvxopt_solvers.options['feastol'] = 1e-10

# define hyper parameters
feature_num = 2
sample_num = 40
batch_size = 10

# define dataset
raw_data = torch.normal(mean=0, std=1, size=(batch_size, feature_num))
cluster1 = raw_data + torch.tensor([2, 4] + ([1] * (feature_num - 2))) # first cluster of data centered at (2, 4)
cluster2 = raw_data + torch.tensor([5, 2] + ([1] * (feature_num - 2))) # second cluster data center at (5, 2)
cluster3 = raw_data + torch.tensor([4, 4] + ([1] * (feature_num - 2))) # third cluster centerred at (4, 4)
cluster4 = raw_data + torch.tensor([5, 4] + ([1] * (feature_num - 2))) # third cluster centerred at (5, 4)

feature_set = torch.cat([cluster1, cluster2, cluster3, cluster4])
label_set = torch.tensor([1] * (batch_size * 2) + [-1] * (batch_size * 2))

torch.save(feature_set, 'SVM/svm_features.log')
torch.save(label_set, 'SVM/svm_labels.log')

# define dual problem parameters, calculate vector a
# P = feature_set * label_set.reshape((-1, 1))
# P = torch.mm(P, P.T).numpy().astype('float')

dot_result = torch.mm(feature_set, feature_set.T)
P = label_set * dot_result * label_set.reshape((-1, 1))
P = P.numpy().astype('float') + np.diag([0.0001] * sample_num)

P = cvxopt_matrix(P)
q = cvxopt_matrix(-np.ones((sample_num, 1)))
G = cvxopt_matrix(-np.eye(sample_num))
h = cvxopt_matrix(np.zeros(sample_num))
A = cvxopt_matrix(label_set.numpy().reshape((-1, 1)).T.astype('float'))
b = cvxopt_matrix(np.zeros(1))

sol = cvxopt_solvers.qp(P, q, G, h, A, b)
ans = np.array(sol['x'])
print(f"later ans = {ans}")
print(f"zero check = {np.dot(A, ans)}")
W = feature_set * label_set.reshape((-1, 1)) * ans.reshape((-1, 1))
W = W.T.sum(dim=1)
b = label_set - (feature_set * W).sum(dim=1)
b = b.mean()
print(f"W = {W}, b = {b}")
print(f"output results: {torch.mv(feature_set.type(torch.float64), W) + b}")


# printing seperating line on graph
plt.scatter(cluster1[:, 0].tolist(), cluster1[:, 1].tolist(), color='blue')
plt.scatter(cluster2[:, 0].tolist(), cluster2[:, 1].tolist(), color='blue')
plt.scatter(cluster3[:, 0].tolist(), cluster3[:, 1].tolist(), color='red')
plt.scatter(cluster4[:, 0].tolist(), cluster4[:, 1].tolist(), color='red')
x = np.linspace(0, 15, 100)
plt.plot(x, (-b - W[0]*x)/W[1])
plt.show()