from mxnet.ndarray.gen_op import col2im
from torch.autograd import backward, grad_mode
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
content_weight, style_weight, tv_weight = 1, 1e3, 10
epoch_num = 500
image_shape = (150, 225)
learning_rate = 0.01

# get pretrained model
vgg19 = models.vgg19(pretrained=True).to(device)

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
style_image = torch.tensor(style_image_[0].asnumpy()).to(device)
content_image_ = nd.load('neural_network/content_img')
content_image = torch.tensor(content_image_[0].asnumpy()).to(device)

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

def extract_feature(image, style_layers, content_layers):
  contents = []
  styles = []
  X = image
  for i in range(0, len(net)):
    X = net[i](X)
    if i in style_layers:
      X_copy = X.clone()
      styles.append(X_copy) 
      # styles.append(X)
    if i in content_layers:
      X_copy = X.clone()
      contents.append(X_copy) 
      # contents.append(X)
      # print(contents)
  
  return styles, contents

# def extract_feature(X, layers):
#   features = []
#   # X = image
#   for i in range(29):
#     X = vgg19.features[i](X)
#     if i in layers:
#       # X_copy = X.detach().clone()
#       # styles.append(X_copy) 
#       features.append(X)
#       break
  
#   return features

# define trainer and loss functions
def gram(X):
  channel_num, n  =  X.shape[1], X.numel() // X.shape[1]
  X = X.view((channel_num, n))
  return torch.mm(X, X.T) / (channel_num * n)

# def gram(tensor):
#     _, d, h, w = tensor.size()
#     tensor = tensor.view(d, h * w)
#     gram = torch.mm(tensor, tensor.t()) / (d * h * w)
#     return gram 

# def content_loss(content_Y_hat, content_Y):
#     return (content_Y_hat - content_Y).square().mean()
  
def content_loss(gen_feat,orig_feat):
    #calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
    content_l=torch.mean((gen_feat-orig_feat)**2)
    return content_l

# def style_loss(style_Y_hat, style_Y_gram):
#   return (gram(style_Y_hat) - style_Y_gram).square().mean()

def style_loss(gen,style):
    #Calculating the gram matrix for the style and the generated image
    batch_size,channel,height,width=gen.shape

    G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t()) / (channel * height * width)
    # A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())

    #Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
    style_l=torch.mean((G-style)**2)
    return style_l

def tv_loss(Y_hat):
  return 0.5 * ((Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).abs().mean() +
                (Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).abs().mean())

def loss(contents_hat, src_content, styles_Y_hat, styles_Y_gram, image):
  
  # contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
  #                contents_hat, src_content)]

  contents_l=0
  for Y_hat, Y in zip(contents_hat, src_content):
    contents_l += (Y_hat - Y).square().mean()

  # style_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip( styles_Y_hat, styles_Y_gram)]
  
  style_l=0
  for Y_hat, Y in zip( styles_Y_hat, styles_Y_gram):
    _, d, h, w = Y_hat.shape
    style_l += (gram(Y_hat) - Y).square().mean()
  
  tv_l = tv_loss(image) * tv_weight
  
  return contents_l, style_l, tv_l

trainer = optim.Adam([image], lr=learning_rate)
# trainer = optim.SGD(image.parameters(), lr=learning_rate)

# define loss from style and content image
src_style, _ = extract_feature(style_image, style_layers, content_layers)
_, src_content = extract_feature(content_image, style_layers, content_layers)

style_Y_gram = [gram(Y) for Y in src_style]
# style_Y_gram = src_style

show_tensor_image(un_normalize_image(image.reshape((3,150,225)), image_mean, image_std))
# show_tensor_image(image.reshape((3,150,225)))

# training
for i in range(epoch_num):
  style_hat, content_hat = extract_feature(image, style_layers, content_layers)
  # style_hat, content_hat = features[:-2] + features[-1:], features[-2:-1]
  c_loss, s_loss, tv_l = loss(content_hat, src_content, style_hat, style_Y_gram, image)
  
  # c_loss=0
  # for Y_hat, Y in zip(content_hat, src_content):
  #   c_loss += (Y_hat - Y).square().mean()

  # c_loss = torch.mean((content_hat[-1] - src_content[-1])**2)
  c_loss *= content_weight

  # s_loss=0
  # for Y_hat, Y in zip(style_hat, style_Y_gram):
  #   _, d, h, w = Y_hat.shape
  #   s_loss += (gram(Y_hat) - Y).square().mean() / (d * h * w)

  # s_loss = 0
  # for layer in range(0, 5):
  #     target_feature = style_hat[layer]
  #     target_gram = gram(target_feature)
  #     _, d, h, w = target_feature.shape
  #     style_gram = style_Y_gram[layer]
  #     # layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
  #     layer_style_loss = torch.mean((target_gram - style_gram)**2)
  #     s_loss += layer_style_loss / (d * h * w)
  s_loss *= style_weight
       
  
  # tv_l = tv_loss(image) * tv_weight
  
  trainer.zero_grad()
  tot_loss = c_loss + s_loss + tv_l 
  # tot_loss.backward()
  if i == (epoch_num - 1):
    tot_loss.backward()
  else:
    tot_loss.backward(retain_graph=True)
  trainer.step()
  if (i % 10) == 0: 
    print(f"epoch {i} loss: {c_loss}, {s_loss}, {tv_l}, total: {tot_loss}")
    # print(f"epoch {i} loss: total: {l}")    

# output 
print("======== finish training =======")
torch.save(image, 'parameter_log/style_transfer.log')
print('data saved in parameter_log/style_transfer.log')

show_tensor_image(un_normalize_image(image.reshape((3,150,225)), image_mean, image_std))
# show_tensor_image(image.reshape((3, 150, 225)))
