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

P = np.array([[8, 0], [0, 12]], dtype='float')
q = np.array([-3, -2], dtype='float')
G = np.array([[-1, 1], [0, 0]], dtype='float')
h = np.array([-1,0], dtype='float')

ans = quadprog_solve_qp(P, q, G, h)
print(ans)
print(f"zero check = {np.dot(ans.T, )}")