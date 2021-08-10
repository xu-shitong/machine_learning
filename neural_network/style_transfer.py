from _typeshed import Self
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn 
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms

# define device used
if torch.cuda.is_available():
  dev = 'cuda:0'
else:
  dev = 'cpu'

device = torch.device(dev)

# define super parameters
content_weight, style_weight, tv_weight = 1, 1e3, 5
epoch_num = 2
image_shape = (1800, 1024)
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
style_image = transform(Image.open('/Users/xushitong/Desktop/整合/背景/en-route.jpg')).reshape((1, 3, image_shape[0], image_shape[1]))
style_image = style_image.to(device)
content_image = transform(Image.open('/Users/xushitong/Desktop/整合/背景/moskou.jpg')).reshape((1, 3, image_shape[0], image_shape[1]))
content_image = content_image.to(device)

# define output image
class GeneratedImage(nn.Module):
  def __init__(self):
    super().__init__()
    self.param = nn.Linear(3, image_shape[0] * image_shape[1])
  def forward(self):
    return self.param.weight.reshape((3, image_shape[0], image_shape[1]))

image = GeneratedImage()

# define structure of model
style_layers = [0, 5, 10, 19, 20]
content_layers = [25]
net = nn.Sequential()
for i in range(max(style_layers + content_layers) + 1):
  net.add_module(f'layer {i}', vgg19.features[i])
net.to(device)

def extract_feature(net, X, style_layers, content_layers):
  style_outputs = []
  content_outputs = []
  for i, layer in enumerate(net):
    X = layer(X)
    if i in style_layers:
      style_outputs.append(X)
    if i in content_layers:
      content_outputs.append(X)
  return X, style_outputs, content_outputs

# define trainer and loss functions
def loss(content_hat, src_content, style_hat, src_style, image, weights):
  assert len(style_hat) == len(src_style)
  assert len(content_hat) == len(src_content)

  def mean_square_loss(y_hat, y):
    return ((y - y_hat) ** 2).mean()

  def gram(X):
    X = X.reshape((X.shape[1], -1))
    return torch.mm(X, X.T) / X.numel()

  # calculate content loss
  content_loss = 0
  for layer_index in range(len(src_content)):
    content_loss += mean_square_loss(content_hat[layer_index], src_content[layer_index])
  print(f"content_loss: {content_loss}")

  # calculate style loss
  style_loss = 0
  for layer_index in range(len(style_hat)):
    X_hat = style_hat[layer_index]
    X = src_style[layer_index]
    style_loss += mean_square_loss(gram(X), gram(X_hat))
  print(f"style_loss: {style_loss}")

  # calculate tv loss
  tv_loss = abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean() + abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean()
  print(f"tv_loss: {tv_loss}")
  
  # weighted sum of the three loss
  return weights[0] * content_loss + weights[1] * style_loss + weights[2] * tv_loss

trainer = optim.Adam(image.parameters(), lr=learning_rate)

# define loss from style and content image
_, src_style, _ = extract_feature(net=net, X=style_image, style_layers=style_layers, content_layers=content_layers)
_, _, src_content = extract_feature(net=net, X=content_image, style_layers=style_layers, content_layers=content_layers)

# training
for i in range(epoch_num):
  Y, style_hat, content_hat = extract_feature(net=net, X=image, style_layers=style_layers, content_layers=content_layers)
  l = loss(content_hat, src_content, style_hat, src_style, image, (content_weight, style_weight, tv_weight))
  l.backward()
  trainer.step()
  print(f"epoch {i} loss: {l}")

# output 
print("======== finishi training =======")
torch.save(image, 'parameter_log/style_transfer.log')
print('data saved in parameter_log/style_transfer.log')
