import torch
from torch import nn

def attention(Q, K, V):
  return torch.sum(torch.softmax(torch.mm(Q, K.T), dim=1) * V, dim=1)

class MultiheadAttention(nn.Module):
  def __init__(self, latent_dim, input_dim, head_num) -> None:
    super().__init__()
    self.Ks = nn.ModuleList([nn.Linear(input_dim, latent_dim // head_num) for _ in range(head_num)])
    self.Qs = nn.ModuleList([nn.Linear(input_dim, latent_dim // head_num) for _ in range(head_num)])
    self.Vs = nn.ModuleList([nn.Linear(input_dim, latent_dim // head_num, bias=False) for _ in range(head_num)])
    self.K_pre = nn.Linear(input_dim, head_num * input_dim)
    self.Q_pre = nn.Linear(input_dim, head_num * input_dim)
    self.V_pre = nn.Linear(input_dim, head_num * input_dim)
    self.final = nn.Linear(latent_dim, latent_dim)

  def forward(self, x):
    for K, Q, V in zip(self.Ks, self.Qs, self.Vs):
      attention(K(x))


class Encoder(nn.Module):
  def __init__(self) -> None:
      super().__init__()
      