import torch
from torch import nn

class EncoderLayer(nn.Module):
  def __init__(self, layer_num, latent_dim, input_dim, head_num) -> None:
    super().__init__()
    self.Ks = nn.Linear(input_dim, head_num * latent_dim)

  def forward(self, pre_k, pre_q, pre_v):



class Encoder(nn.Module):
  def __init__(self) -> None:
      super().__init__()
      