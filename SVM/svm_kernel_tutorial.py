import torch
import matplotlib.pyplot as plt
import numpy as np
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
epsilon = 1e-2 # define range of value classified as 0, used in plotting session

# define dataset
raw_data = torch.normal(mean=0, std=1, size=(batch_size, feature_num))
cluster1 = raw_data + torch.tensor([2, 4] + ([1] * (feature_num - 2))) # first cluster of data centered at (2, 4)
cluster2 = raw_data + torch.tensor([5, 2] + ([1] * (feature_num - 2))) # second cluster data center at (5, 2)
cluster3 = raw_data + torch.tensor([4, 4] + ([1] * (feature_num - 2))) # third cluster centerred at (4, 4)
cluster4 = raw_data + torch.tensor([5, 4] + ([1] * (feature_num - 2))) # third cluster centerred at (5, 4)

feature_set = torch.cat([cluster1, cluster2, cluster3, cluster4])
label_set = torch.tensor([1] * (batch_size * 2) + [-1] * (batch_size * 2))

X = feature_set.type(torch.float64).numpy()
y = label_set.type(torch.float64).reshape((-1, 1)).numpy()

# define dual problem parameters, calculate vector alpha
# dual problem: 
#   minimize: alpha^T * P * alpha + q^T * a
#   subject to: for all alpha[i] >= 0
#               y^T * alpha = 0
P = np.matmul(y, y.T) * np.dot(X, X.T)**2 + np.eye(sample_num) * 0.0001
P = cvxopt_matrix(P) 
q = cvxopt_matrix(-np.ones((sample_num, 1)))
G = cvxopt_matrix(-np.eye(sample_num))
h = cvxopt_matrix(np.zeros(sample_num))
A = cvxopt_matrix(y.reshape((1, -1)))
b = cvxopt_matrix(np.zeros(1))

# solve dual problem
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
ans = np.array(sol['x'])

# get samples that corrisponds to the alpha with positive value
ind = (ans > 1e-4).flatten()
sv = X[ind]
sv_y = y[ind]
alphas = ans[ind]

print(f"raw alpha = {ans}") # alpha without clamp only positive value
print(f"zero check = {np.dot(A, ans)}") # check y^T * alpha = 0 condition is met

# get bias value
b = sv_y - np.sum(np.dot(sv, sv.T)**2 * sv_y * alphas, axis=0)
b = np.sum(b) / b.size

# get coordinates where SVM classify as 0, these points lie on the boundary SVM use
test_matrix = []
boundaries_i = []
boundaries_j = []
for i in np.linspace(0, 15, 100):
  for j in np.linspace(0, 15, 100):
    prod = np.sum(np.dot(sv, np.array([[i, j]]).T)**2 * alphas * sv_y, axis=0) + b
    if -epsilon < prod.item() < epsilon:
      boundaries_i.append(i)
      boundaries_j.append(j)
print(f"boundary length = {len(boundaries_i)}")

# printing samples and boundary 
plt.scatter(feature_set[:(batch_size*2), 0].tolist(), feature_set[:(batch_size*2), 1].tolist(), color='blue')
plt.scatter(feature_set[(batch_size*2):, 0].tolist(), feature_set[(batch_size*2):, 1].tolist(), color='red')
plt.scatter(boundaries_i, boundaries_j, color='black')
plt.show()

