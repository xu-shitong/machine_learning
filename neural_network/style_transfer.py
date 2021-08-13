from torch.nn import parameter
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn 
import torch.optim as optim
from PIL import Image
import sys
sys.path.append('../machine_learning/')
from utils.functions import show_tensor_image, un_normalize_image
import mxnet
from mxnet import nd

# define device used
if torch.cuda.is_available():
  dev = 'cuda:0'
else:
  dev = 'cpu'

device = torch.device(dev)

# define super parameters
content_weight, style_weight, tv_weight = 8, 70, 1
epoch_num = 51
image_shape = (150, 225)
learning_rate = 0.004

# get pretrained model
# vgg19 = models.vgg19(pretrained=True)

# get style and content image
image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])

transform = transforms.Compose([
  transforms.Resize(image_shape),
  transforms.ToTensor(),
  transforms.Normalize(image_mean, image_std)
])
# style_image = transform(Image.open('training_data/images/autumn_oak.jpg')).reshape((1, 3, image_shape[0], image_shape[1]))
# style_image = style_image.to(device)
# content_image = transform(Image.open('training_data/images/rainier.jpg')).reshape((1, 3, image_shape[0], image_shape[1]))
# content_image = content_image.to(device)
style_image_ = nd.load('neural_network/style_img')
style_image = torch.tensor(style_image_[0].asnumpy())
content_image_ = nd.load('neural_network/content_img')
content_image = torch.tensor(content_image_[0].asnumpy())

# define output image
# class GeneratedImage(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.param = nn.Linear(image_shape[0] * image_shape[1], 3)
#     with torch.no_grad():
#       content_image_copy = content_image.detach().clone()
#       self.param.weight.copy_(content_image_copy.reshape((3, image_shape[0] * image_shape[1])))
#   def forward(self):
#     return self.param.weight.reshape((1, 3, image_shape[0], image_shape[1]))

# image = GeneratedImage()
# image = content_image.detach().clone()
# image.requires_grad = True
image=content_image.clone().requires_grad_(True)
image.to(device)

# define structure of model
style_layers = [0, 5, 10, 19, 28]
content_layers = [25]
# net = nn.Sequential()
# for i in range(max(style_layers + content_layers) + 1):
#   net.add_module(f'layer {i}', vgg19.features[i])
net = torch.load('neural_network/param.log')
net.to(device)

def extract_feature(X, style_layers, content_layers):
  contents = []
  styles = []
  for i in range(0, len(net)):
    X = net[i](X)
    if i in style_layers:
      X_copy = X.detach().clone()
      styles.append(X_copy) 
    if i in content_layers:
      X_copy = X.detach().clone()
      contents.append(X_copy) 
  
  return styles, contents

# define trainer and loss functions
def gram(X):
  channel_num, n  =  X.shape[1], X.numel() // X.shape[1]
  X = X.reshape((channel_num, n))
  return torch.mm(X, X.T) / (channel_num * n)

def content_loss(content_Y_hat, content_Y):
    return (content_Y_hat - content_Y).square().mean()
  
def style_loss(style_Y_hat, style_Y_gram):
  return (gram(style_Y_hat) - style_Y_gram).square().mean()

def tv_loss(Y_hat):
  return 0.5 * ((Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).abs().mean() +
                (Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).abs().mean())

def loss(contents_hat, src_content, styles_Y_hat, styles_Y_gram, image):
  
  contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
                 contents_hat, src_content)]
  
  style_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip( styles_Y_hat, styles_Y_gram)]
  
  tv_l = tv_loss(image) * tv_weight
  
  tot_loss = tv_l
  for l in contents_l:
    tot_loss = tot_loss.add(l)
  for l in style_l:
    tot_loss = tot_loss.add(l)

  return sum(contents_l), sum(style_l), tv_l, tot_loss

trainer = optim.Adam([image], lr=learning_rate)
# trainer = optim.SGD(image.parameters(), lr=learning_rate)

# define loss from style and content image
src_style, _ = extract_feature(X=style_image, style_layers=style_layers, content_layers=content_layers)
_, src_content = extract_feature(X=content_image, style_layers=style_layers, content_layers=content_layers)

style_Y_gram = [gram(Y) for Y in src_style]

show_tensor_image(un_normalize_image(image.reshape((3,150,225)), image_mean, image_std))
# show_tensor_image(image.reshape((3,150,225)))

# training
for i in range(epoch_num):
  style_hat, content_hat = extract_feature(X=image, style_layers=style_layers, content_layers=content_layers)
  c_loss, s_loss, tv_l, l = loss(content_hat, src_content, style_hat, style_Y_gram, image)
  trainer.zero_grad()
  if i == (epoch_num - 1):
    l.backward()
  else:
    l.backward(retain_graph=True)
  trainer.step()
  if (i % 10) == 0: 
    print(f"epoch {i} loss: {c_loss}, {s_loss}, {tv_l}, total: {l}")

# output 
print("======== finish training =======")
torch.save(image, 'parameter_log/style_transfer.log')
print('data saved in parameter_log/style_transfer.log')

show_tensor_image(un_normalize_image(image.reshape((3,150,225)), image_mean, image_std))
# show_tensor_image(image.reshape((3, 150, 225)))