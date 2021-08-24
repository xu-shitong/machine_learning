import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import quadprog

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    # qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_G = .5 * P   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

# define hyper parameters
feature_num = 2
sample_num = 40
batch_size = 10

# define dataset
raw_data = torch.normal(mean=0, std=0.5, size=(batch_size, feature_num))
cluster1 = raw_data + torch.tensor([2, 4] + ([1] * (feature_num - 2))) # first cluster of data centered at (2, 4)
cluster2 = raw_data + torch.tensor([5, 2] + ([1] * (feature_num - 2))) # second cluster data center at (5, 2)
cluster3 = raw_data + torch.tensor([4, 8] + ([1] * (feature_num - 2))) # third cluster centerred at (4, 4)
cluster4 = raw_data + torch.tensor([10, 4] + ([1] * (feature_num - 2))) # third cluster centerred at (5, 4)

feature_set = torch.cat([cluster1, cluster2, cluster3, cluster4])
label_set = torch.tensor([1] * (batch_size * 2) + [-1] * (batch_size * 2))

# visualize data
plt.scatter(cluster1[:, 0].tolist(), cluster1[:, 1].tolist(), color='blue')
plt.scatter(cluster2[:, 0].tolist(), cluster2[:, 1].tolist(), color='blue')
plt.scatter(cluster3[:, 0].tolist(), cluster3[:, 1].tolist(), color='red')
plt.scatter(cluster4[:, 0].tolist(), cluster4[:, 1].tolist(), color='red')
plt.show()

torch.save(feature_set, 'SVM/svm_features.log')
torch.save(label_set, 'SVM/svm_labels.log')

# # read data from defined dataset
# feature_set = torch.load('SVM/svm_features.log')
# label_set = torch.load('SVM/svm_labels.log')

# # visualize dataset
# plt.scatter(feature_set[:, 0].tolist(), feature_set[:, 1].tolist(), 1)
# plt.show()

# define dual problem parameters, calculate vector a
P = feature_set * label_set.reshape((-1, 1))
P = torch.mm(P, P.T).numpy().astype('float')
P += np.diag([0.0001] * sample_num)
q = -np.ones(sample_num).astype('float')

G = -np.diag([1] * sample_num).astype('float')
h = np.zeros(sample_num).astype('float')
ans = quadprog_solve_qp(P, q, G, h)
print(f"ans = {ans}")
print(f"features = {feature_set}")
print(f"label = {label_set}")
print(f"P = {P}")
print(f"q = {q}")
print(f"G = {G}")
print(f"h = {h}")
W = feature_set * label_set.reshape((-1, 1)) * ans.reshape((-1, 1))
W = W.T.sum(dim=1)
b = label_set - (feature_set * W).sum(dim=1)
b = b.mean()
print(f"W = {W}, b = {b}")
print(f"output results: {torch.mv(feature_set.type(torch.float64), W) + b}")
