import torch
from torch import nn
from d2l import torch as d2l
# define hyper parameter
sizes = [] # ratio of anchor boxes relative to the tensor width or height
ratios = [] # ratio of height and width of anchor used
anchor_num = len(sizes) + len(ratios) - 1
class_num = 10

# get dataset

# define network
def predict(layer, cls_predictor, box_predictor, X):
  Y = layer(X)
  anchors = d2l.multibox_prior(Y, sizes, ratios)
  return Y, anchors, cls_predictor(Y), box_predictor(Y)

# flattern a list of lists, used in concating result from different blocks
def list_flatten(list):
  return [ e for elem in list for e in elem]

# define class predictor convolution layer
def get_cls_predictor(in_channels):
  return nn.Conv2d(in_channels=in_channels, out_channels=anchor_num * (class_num + 1), kernel_size=(3, 3), padding=1)

# define anchor boundary box offset predict layer
def get_box_predictor(in_channels):
  return nn.Conv2d(in_channels=in_channels, out_channels=anchor_num * 4, kernel_size=(3, 3), padding=1)

class hw_half_blk(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
      nn.BatchNorm2d(num_features=out_channels),
      nn.ReLU(inplace=True),

      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
      nn.BatchNorm2d(num_features=out_channels),
      nn.ReLU(inplace=True),

      nn.MaxPool2d(kernel_size=(2, 2))
    )

  def forward(self, X):
    return self.block(X)
    
class SSD(nn.Module):
  def __init__(self, channel_nums):
    super().__init__()
    self.layers = [
      # 1.basic net block
      nn.Sequential(
        hw_half_blk(channel_nums[0], channel_nums[1]),
        hw_half_blk(channel_nums[1], channel_nums[2]),
        hw_half_blk(channel_nums[2], channel_nums[3])
      ),
      # 2-4.height width half reduce block
      hw_half_blk(channel_nums[3], channel_nums[4]),
      hw_half_blk(channel_nums[4], channel_nums[5]),
      hw_half_blk(channel_nums[5], channel_nums[6]),

      # 5.global maximum block
      nn.AdaptiveMaxPool2d(1)]
    self.cls_predictors = [get_cls_predictor(channel_nums[i]) for i in range(3, 7)] + \
                          [get_cls_predictor(1)]
    self.box_predictors = [get_box_predictor(channel_nums[i]) for i in range(3, 7)] + \
                          [get_box_predictor(1)]

  def forward(self, X):
    anchors, cls_predicts, box_predicts = [None] * 5, [None] * 5, [None] * 5
    for i, layer in enumerate(self.layers):
      X, anchors[i], cls_predicts[i], box_predicts[i] = predict(layer, self.cls_predictors[i], self.box_predictors[i], X)
    return X, list_flatten(anchors), list_flatten(cls_predicts), list_flatten(box_predicts)

net = SSD([3, 16, 32, 64, 128, 128, 128])

# 


