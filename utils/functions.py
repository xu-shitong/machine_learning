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