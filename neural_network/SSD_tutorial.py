# WIP: This code has not been finished, require further work on multiBoxTarget, data reading and iterating implemented

import torch
from torch import nn
from d2l import torch as d2l
from torch import optim

# define hyper parameter
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]] # ratio of anchor boxes relative to the tensor width or height
ratios = [[1, 2, 0.5]] * 5 # ratio of height and width of anchor used
anchor_num = len(sizes) + len(ratios) - 1
class_num = 10
epoch_num = 3

# get dataset


# define network
def predict(layer, cls_predictor, box_predictor, sizes, ratios, X):
  Y = layer(X)
  anchors = d2l.multibox_prior(Y, sizes, ratios)
  return Y, anchors, cls_predictor(Y), box_predictor(Y)

# flatten output from one layer
def single_layer_flatten(layer):
  # reshape to (batch_size, height, width, channel_num)
  print(f"shape = {layer.shape}")
  return layer.permute(0, 2, 3, 1).flatten(start_dim=1)

# flattern a list of lists, used in concating result from different blocks
def list_flatten(list):
  return torch.cat([single_layer_flatten(l) for l in list], dim=1)

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
      nn.AdaptiveMaxPool2d(128)]
    self.cls_predictors = [get_cls_predictor(channel_nums[i]) for i in range(3, 7)] + \
                          [get_cls_predictor(128)]
    self.box_predictors = [get_box_predictor(channel_nums[i]) for i in range(3, 7)] + \
                          [get_box_predictor(128)]

  def forward(self, X):
    anchors, cls_predicts, box_predicts = [None] * 5, [None] * 5, [None] * 5
    for i, layer in enumerate(self.layers):
      print(f"computing layer {i}'s output")
      X, anchors[i], cls_predicts[i], box_predicts[i] = predict(layer, self.cls_predictors[i], self.box_predictors[i], sizes[i], ratios[i], X)
    return X, torch.cat(anchors, dim=1), list_flatten(cls_predicts), list_flatten(box_predicts)

net = SSD([3, 16, 32, 64, 128, 128, 128])

# # trial on forward on a same input data
# X = torch.ones((32, 3, 256, 256))

# Y, anchors, cls_preds, bbox_preds = net(X)
# print(f"cls_preds = {cls_preds}")
# print(f"bbox_preds = {bbox_preds}")

# define loss function


# train
trainer = optim.SGD(net.get_parameter(), lr=0.01)


for i in range(epoch_num):
  acc_loss = 0
  for img, target_anchors in data_iter:
    Y_hat, anchors, cls_predicts, box_predicts = net(img)
    l = loss(anchors, cls_predicts, box_predicts)
    trainer.zero_grad()
    l.backward()
    trainer.step()

    acc_loss += l
  print(f"finish epoch {i} with acc loss: {acc_loss}")

print(f"============= finish training ============")
torch.save(net, 'parameter_log/SSD.log')
print(f"parameter saved in parameter_log/SSD.log")
