import torch
from torch import nn

def attention(Q, K, V):
  return torch.sum(torch.softmax(torch.mm(Q, K.T), dim=1) * V, dim=1)

class MultiheadAttention(nn.Module):
  def __init__(self, time_dim, latent_dim, d_ff, head_num) -> None:
    super().__init__()
    self.Ks = nn.ModuleList([nn.Linear(time_dim, latent_dim // head_num, bias=False) for _ in range(head_num)])
    self.Qs = nn.ModuleList([nn.Linear(time_dim, latent_dim // head_num, bias=False) for _ in range(head_num)])
    self.Vs = nn.ModuleList([nn.Linear(time_dim, latent_dim // head_num, bias=False) for _ in range(head_num)])
    self.K_pre = nn.Linear(latent_dim, latent_dim)
    self.Q_pre = nn.Linear(latent_dim, latent_dim)
    self.V_pre = nn.Linear(latent_dim, latent_dim)
    self.final = nn.Linear(latent_dim, latent_dim)

  def forward(self, x_k, x_q, x_v):
    k, q, v = self.K_pre(x_k), self.Q_pre(x_q), self.V_pre(x_v)
    for K, Q, V in zip(self.Ks, self.Qs, self.Vs):
      attention(K(k), Q(q), V(v))


class Encoder(nn.Module):
  def __init__(self) -> None:
      super().__init__()
      