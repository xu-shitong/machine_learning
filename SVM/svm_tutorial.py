import torch
import matplotlib.pyplot as plt
import quadprog
import numpy as np

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
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
sample_num = 6

# # define dataset
# raw_data = torch.normal(mean=2, std=0.5, size=(sample_num, feature_num))
# cluster1 = raw_data + torch.tensor([2, 4]) # first cluster of data centered at (2, 4)
# cluster2 = raw_data + torch.tensor([5, 2]) # second cluster data center at (5, 2)
# cluster3 = raw_data + torch.tensor([4, 4]) # third cluster centerred at (4, 4)

# feature_set = torch.cat([cluster1, cluster2, cluster3])
# label_set = torch.tensor([1] * sample_num * 2 + [-1] * sample_num)


# # visualize data
# plt.scatter(cluster1[:, 0].tolist(), cluster1[:, 1].tolist(), 1)
# plt.scatter(cluster2[:, 0].tolist(), cluster2[:, 1].tolist(), 1)
# plt.scatter(cluster3[:, 0].tolist(), cluster3[:, 1].tolist(), 2)
# plt.show()

# torch.save(feature_set, 'svm_features.log')
# torch.save(label_set, 'svm_labels.log')

# read data from defined dataset
feature_set = torch.load('SVM/svm_features.log')
label_set = torch.load('SVM/svm_labels.log')

# visualize dataset
# plt.scatter(feature_set[:, 0].tolist(), feature_set[:, 1].tolist(), 1)
# plt.show()

# define QP equation parameters, calculate vector a
G = np.zeros((sample_num, sample_num))
for i in range(sample_num):
  for j in range(sample_num):
    G[i, j] = label_set[i] * label_set[j] * sum(feature_set[i] * feature_set[j])
a = -np.ones((sample_num, ))

C = -np.diag(np.ones((sample_num, )))
b = np.zeros((sample_num, ))

print(f"G = {G}")
print(f"a = {a}")
print(f"C = {C}")
print(f"b = {b}")

ans = quadprog_solve_qp(G, a, C, b) # parameter passed in wrong??
print(ans)