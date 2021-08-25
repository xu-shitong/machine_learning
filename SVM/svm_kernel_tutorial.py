import torch
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

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

# define dual problem parameters, calculate vector a
dot_result = torch.mm(feature_set, feature_set.T)**2
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
ind = (ans > 1e-4).flatten()
sv = feature_set[ind].numpy()
sv_y = label_set[ind].numpy()
alphas = ans[ind]

print(f"later ans = {ans}")
print(f"zero check = {np.dot(A, ans)}")

# print(f"svy = {sv_y}")
# b = sv_y - (torch.mm(sv, sv.T)**2 * alphas * sv_y).sum()
# b = b.sum() / b.numpy().size

# v = alphas.reshape((-1, 1)) * label_set.reshape((-1, 1)).numpy()
# v = v.type(torch.float32)

# b = label_set - (v * torch.mm(feature_set, feature_set.T)).sum(dim=1)
# b = (b * map).sum() / map.sum()
# print(f"b = {b}")
# print(f"alpha = {alphas}, sv_y = {sv_y}, sv = {sv}")
# weighted = np.dot(alphas.reshape((1, -1)) * sv_y.numpy(), (torch.matmul(sv, feature_set.T)**2).numpy())
# print(weighted)
# result = weighted + b.numpy()
# print(result)
# # printing seperating line on graph
# plt.scatter(cluster1[:, 0].tolist(), cluster1[:, 1].tolist(), color='blue')
# plt.scatter(cluster2[:, 0].tolist(), cluster2[:, 1].tolist(), color='blue')
# plt.scatter(cluster3[:, 0].tolist(), cluster3[:, 1].tolist(), color='red')
# plt.scatter(cluster4[:, 0].tolist(), cluster4[:, 1].tolist(), color='red')

# boundaries = []
# for i in np.linspace(0, 15, 100):
#   for j in np.linspace(0, 15, 100):
#     weighted = np.dot(v.T, torch.matmul(feature_set, torch.tensor([i, j]).type(torch.float32))**2)
#     result = weighted + b.numpy()
#     boundaries.append(result)

# plt.plot(boundaries, color='black')
# plt.show()



def poly_kernel(x, z, degree, intercept):
        return np.power(np.matmul(x, z.T) + intercept, degree)

b = sv_y - np.sum(poly_kernel(sv, sv, degree = 2, intercept = 0) * alphas * sv_y, axis=0)
b = np.sum(b) / b.size
print(f"b = {b}")
prod = np.sum(poly_kernel(sv, feature_set.numpy(), degree = 2, intercept = 0) * alphas.reshape((-1,1)) * sv_y.reshape((-1,1)), axis=0)
print(prod)
prod += b
print(prod)
