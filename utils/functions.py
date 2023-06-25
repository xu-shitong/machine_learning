from matplotlib import pyplot as plt
import torch
from torchvision import transforms

# carry out EPOCH_NUM times
# print out FST_DISPLAY_NUM loss values in the first batch
def train(net, data_iter, loss, trainer, epoch_num, fst_display_num=5):
  batch_index = 0 # only used for printing first FST_DISPAY_NUM loss
  data_length = len(data_iter) # number of batches, used for calculating mean batch loss
  for i in range(1, epoch_num+1):
    acc_loss = 0
    for X, y in data_iter:
      y_hat = net(X)
      l = loss(y_hat, y)
      trainer.zero_grad()
      l.backward()
      trainer.step()
      acc_loss += l
      if i == 0 and batch_index < fst_display_num: 
        print(f"epoch 1 loss+ {l}")
        batch_index += 1
    print(f"epoch {i} has acc loss: {acc_loss / data_length}")

  print("======= training finished ========")
  for param in list(net.parameters()):
    print(f"paramname: {param.name} has parameter {param.data}")

# display image from its tensor representation
def show_tensor_image(image_tensor):
  image_tensor = image_tensor.clamp(min=0, max=1)
  image = transforms.ToPILImage()(image_tensor)
  image.show()

# apply unnormalize on image tensor with given rgb_mean and rgb_stdz
# rgb_mean nad rgb_std are python array with 3 elements
def un_normalize_image(image_tensor, rgb_mean, rgb_std):
  image_tensor = image_tensor.permute((1, 2, 0))
  image_tensor = (image_tensor * rgb_std) + rgb_mean
  return image_tensor.permute((2,0,1))

# add anchor boxes to the fig axes
# anchor_boxes format: [[top left x, top left y, bottom right x, bottom right y], ...]
# anchor_name format: contain the same number of string as anchor box numbers
def apply_anchor_boxes(fig, anchor_boxes, anchor_names=None, box_colour='blue'):
  if anchor_names is not None:
    assert len(anchor_boxes) == len(anchor_names)
  for i, box in enumerate(anchor_boxes):
    fig.axes.add_patch(plt.Rectangle(xy=(box[0], box[1]), fill=False, width=box[2]-box[0], height=box[3]-box[1], linewidth=2, edgecolor=box_colour))
    if anchor_names is not None:
      fig.axes.text(box[0], box[1], anchor_names[i], va='center', ha='center', fontsize=9, color='white', bbox=dict(facecolor='blue', lw=0))

'''
Following 3 functions are used to visualize different groups' learning curves in 2 figures'''
import re
import matplotlib.pyplot as plt
import torch
def extract_float(file_name, start="Epoch"):
  def isfloat(word):
    try:
      float(word)
      return True
    except ValueError:
      return False
  file = open(file_name)
  l = [[float(word) for word in re.split("\n|,| |\t|\(|\)|\[|\]", line) if isfloat(word)] for line in file if line.startswith(start)]
  return torch.tensor(l)

def find_mean_index(data):
  '''
  data shape: [batch size, number of logged metrics]
    the first metric logged must be epoch number
  
  return: list of index indicating mean loss positions
  '''
  last_epoch_num = 0
  indice = []
  for i, e in enumerate(data[:, 0]):
    if e != last_epoch_num:
      indice.append(i-1)
      last_epoch_num = e
  indice.append(i)
  return indice

def show_plot(all_file_names, line_labels, graph_name="", train_index=6, val_index=8, cap=None, start="Epoch", cut=1000, lowercut=0):
  cs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  # fig, axarr = plt.subplots(1, 2, figsize=(15, 4))
  fig, axarr = plt.subplots(1, 2, figsize=(15, 5))

  for i, file_names in enumerate(all_file_names):
    print("plotting cluster {}".format(i))
    data = []
    for file_name in file_names:
        d = extract_float(file_name, start)
        if cap:
            data.append(d.unsqueeze(0)[:, find_mean_index(d)].clip(0, cap))
        else:
            data.append(d.unsqueeze(0)[:, find_mean_index(d)])
        print(f"{re.split('/|_', file_name)[4]}, min train loss: {data[-1].min(dim=1)[0][0, train_index]}, min val loss: {data[-1].min(dim=1)[0][0, val_index]}")

    data = torch.cat(data)[:, lowercut:cut]

    mean_data = data[:, :, train_index].mean(dim=0)
    axarr[0].plot(list(range(lowercut, lowercut + data.shape[1])), mean_data, label=line_labels[i], color=cs[i])
    if data.shape[0] != 1:
        var_data = data[:, :, train_index].std(dim=0)
        axarr[0].fill_between(list(range(lowercut, lowercut + data.shape[1])), mean_data - var_data, mean_data + var_data, alpha=0.2, color=cs[i])
    axarr[0].legend()
    axarr[0].set_title(graph_name + " training curve")
    axarr[0].set(xlabel='Epoch', ylabel='Loss')

    mean_data = data[:, :, val_index].mean(dim=0)
    axarr[1].plot(list(range(lowercut, lowercut + data.shape[1])), mean_data, label=line_labels[i], color=cs[i])
    if data.shape[0] != 1:
        var_data = data[:, :, val_index].std(dim=0)
        axarr[1].fill_between(list(range(lowercut, lowercut + data.shape[1])), mean_data - var_data, mean_data + var_data, alpha=0.2, color=cs[i])
    axarr[1].legend()
    axarr[1].set_title(graph_name + " validation curve")
    axarr[1].set(xlabel='Epoch', ylabel='Loss')

  return fig, axarr