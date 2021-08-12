from torch.functional import norm
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn 
import torch.optim as optim
from PIL import Image

# define device used
if torch.cuda.is_available():
  dev = 'cuda:0'
else:
  dev = 'cpu'

device = torch.device(dev)

# define super parameters
content_weight, style_weight, tv_weight = 1, 1e3, 10
epoch_num = 1
image_shape = (150, 225)
learning_rate = 0.01

# get pretrained model
vgg19 = models.vgg19(pretrained=True)

# get style and content image
image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])

transform = transforms.Compose([
  transforms.Resize(image_shape),
  transforms.ToTensor(),
  transforms.Normalize(image_mean, image_std)
])
style_image = transform(Image.open('training_data/images/autumn_oak.jpg')).reshape((1, 3, image_shape[0], image_shape[1]))
style_image = style_image.to(device)
content_image = transform(Image.open('training_data/images/rainier.jpg')).reshape((1, 3, image_shape[0], image_shape[1]))
content_image = content_image.to(device)

# define output image
# class GeneratedImage(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.param = nn.Linear(image_shape[0] * image_shape[1], 3)
#     with torch.no_grad():
#       self.param.weight.copy_(content_image.reshape((3, image_shape[0] * image_shape[1])))
#   def forward(self):
#     return self.param.weight.reshape((1, 3, image_shape[0], image_shape[1]))

# image = GeneratedImage()
image = content_image.detach().clone()
image.requires_grad = True
image.to(device)

# define structure of model
style_layers = [0, 5, 10, 19, 28]
content_layers = [25]
net = nn.Sequential()
for i in range(max(style_layers + content_layers) + 1):
  net.add_module(f'layer {i}', vgg19.features[i])

net.to(device)

def extract_feature(net, X, style_layers, content_layers):
  # style_outputs = []
  # content_outputs = []
  # for i, layer in enumerate(net):
  #   X = layer(X)
  #   if i in style_layers:
  #     style_outputs.append(X)
  #   if i in content_layers:
  #     content_outputs.append(X)
      
  contents = []
  styles = []
  for i in range(len(net)):
    X = net[i](X)
    if i in style_layers:
      styles.append(X) 
    if i in content_layers:
      contents.append(X) 

  return styles, contents

# define trainer and loss functions
def gram(X):
  channel_num, n  =  X.shape[1], X.numel() // X.shape[1]
  X = X.reshape((channel_num, n))
  return torch.mm(X, X.T) / (channel_num * n)

def loss(contents_hat, src_content, styles_Y_hat, styles_Y_gram, image):
  def content_loss(content_Y_hat, content_Y):
    return (content_Y_hat - content_Y).square().mean()
  
  def style_loss(style_Y_hat, style_Y_gram):
    return (gram(style_Y_hat) - style_Y_gram).square().mean()
  
  def tv_loss(Y_hat):
    return 0.5 * (abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
  
  contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
                 contents_hat, src_content)]
  
  style_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip( styles_Y_hat, styles_Y_gram)]
  
  tv_l = tv_loss(image) * tv_weight
  
  return sum(contents_l), sum(style_l), tv_l

trainer = optim.Adam([image], lr=learning_rate)

# define loss from style and content image
src_style, _ = extract_feature(net=net, X=style_image, style_layers=style_layers, content_layers=content_layers)
_, src_content = extract_feature(net=net, X=content_image, style_layers=style_layers, content_layers=content_layers)

style_Y_gram = [gram(Y) for Y in src_style]

# training
for i in range(epoch_num):
  style_hat, content_hat = extract_feature(net=net, X=image, style_layers=style_layers, content_layers=content_layers)
  c_loss, s_loss, tv_loss = loss(content_hat, src_content, style_hat, style_Y_gram, image)
  l = c_loss + s_loss + tv_loss
  if i == (epoch_num - 1):
    l.backward()
  else:
    l.backward(retain_graph=True)
  trainer.step()
  if (i % 50) == 0: 
    print(f"epoch {i} loss: {c_loss}, {s_loss}, {tv_loss}, total: {l}")

# output 
print("======== finishi training =======")
torch.save(image, 'parameter_log/style_transfer.log')
print('data saved in parameter_log/style_transfer.log')
