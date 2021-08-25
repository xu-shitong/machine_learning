from matplotlib.colors import BoundaryNorm
import numpy as np
from cvxopt import matrix, solvers
import torch
import matplotlib.pyplot as plt
from torch._C import dtype


def poly_kernel(x, z, degree, intercept):
        return np.power(np.matmul(x, z.T) + intercept, degree)

# X = np.array([[4,2], [4,4], [2,4]], dtype='float64')
# y = np.array([[1],[-1],[1]], dtype='float64')

# define hyper parameters
feature_num = 2
sample_num = 40
batch_size = 10

# # define dataset
# raw_data = torch.normal(mean=0, std=1, size=(batch_size, feature_num))
# cluster1 = raw_data + torch.tensor([2, 4] + ([1] * (feature_num - 2))) # first cluster of data centered at (2, 4)
# cluster2 = raw_data + torch.tensor([5, 2] + ([1] * (feature_num - 2))) # second cluster data center at (5, 2)
# cluster3 = raw_data + torch.tensor([4, 6] + ([1] * (feature_num - 2))) # third cluster centerred at (4, 4)
# cluster4 = raw_data + torch.tensor([6, 4] + ([1] * (feature_num - 2))) # third cluster centerred at (5, 4)

# X = torch.cat([cluster1, cluster2, cluster3, cluster4])
# y = torch.tensor([1] * (batch_size * 2) + [-1] * (batch_size * 2)).type(torch.float64)
# y = y.reshape((-1,1))

# # print(X)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

torch.save(X, 'SVM/svm_features.log')
torch.save(y, 'SVM/svm_labels.log')

# X = torch.load('SVM/svm_features.log')
# y = torch.load('SVM/svm_labels.log')

X = X.type(torch.float64).numpy()
y = y.numpy()

m, n = X.shape
K = poly_kernel(X, X, degree = 2, intercept = 1)
P = matrix(np.matmul(y,y.T) * K)
q = matrix(np.ones((m, 1)) * -1)
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros(1))
G = matrix(np.eye(m) * -1)
h = matrix(np.zeros(m))

solution = solvers.qp(P, q, G, h, A, b)
alphas = np.array(solution['x'])
ind = (alphas > 1e-4).flatten()
sv = X[ind]
sv_y = y[ind]
alphas = alphas[ind]
print(f"alphas = {alphas}")

b = sv_y - np.sum(poly_kernel(sv, sv, degree = 2, intercept = 1) * alphas * sv_y, axis=0)
b = np.sum(b) / b.size
print(f"b = {b}")

prod = np.sum(poly_kernel(sv, X, degree = 2, intercept = 1) * alphas * sv_y, axis=0) + b
print(f"prod = {prod}")

# boundaries = []
# all = []
epsilon = 1e-1

# for i in np.linspace(0, 15, 100):
#   for j in np.linspace(0, 15, 100):
#     # result = np.sum(poly_kernel(sv, np.array([i, j], dtype='float64'), degree = 2, intercept = 1) * alphas * sv_y)
#     result = np.sum(poly_kernel(sv, np.array([[i, j]]), degree = 2, intercept = 1) * alphas * sv_y, axis=0)
#     result = result + b
#     all.append([i, j, int(result.item())])
# print(f"result = {result}")
# print(f"min = {min(all)}, max = {max(all)}")

test_matrix = []
boundaries_i = []
boundaries_j = []
for i in np.linspace(0, 15, 100):
  for j in np.linspace(0, 15, 100):
    prod = np.sum(poly_kernel(sv, np.array([[i, j]]), degree = 2, intercept = 1) * alphas * sv_y, axis=0) + b
    if -epsilon < prod.item() < epsilon:
      boundaries_i.append(i)
      boundaries_j.append(j)
print(len(boundaries_i))


# printing seperating line on graph
plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), color='blue')
plt.scatter(boundaries_i, boundaries_j, color='black')
plt.show()