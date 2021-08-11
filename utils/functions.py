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

def show_tensor_image(image_path, image_shape):
  image_ = torch.load(image_path, map_location=torch.device('cpu'))
  image = transforms.ToPILImage()(image_.reshape(image_shape))
  image.show()