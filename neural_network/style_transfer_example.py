from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
from mxnet import nd
import sys
sys.path.append('../machine_learning/')
from utils.functions import show_tensor_image, un_normalize_image


vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)
image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])


style_image_ = nd.load('neural_network/style_img')
style = torch.tensor(style_image_[0].asnumpy())
content_image_ = nd.load('neural_network/content_img')
content = torch.tensor(content_image_[0].asnumpy())

# def get_features(image, model, layers=None):
    
#     if layers is None:
#         layers = {'0': 'conv1_1',
#                   '5': 'conv2_1', 
#                   '10': 'conv3_1', 
#                   '19': 'conv4_1',
#                   '30': 'conv5_2', #content
#                   '28': 'conv5_1'}
        
#     features = {}
#     x = image
#     for name, layer in model._modules.items():
#         x = layer(x)
#         if name in layers:
#             features[layers[name]] = x
            
#     return features

def get_features(image, model, layers=None):
  features = []
  X = image
  if layers is None:
    layers = [0, 5,10,19,28,30] 

  for i in range(31):
    X = model[i](X)
    if i in layers:
      # X_copy = X.detach().clone()
      # styles.append(X_copy) 
      features.append(X)
  
  return features

def gram_matrix(tensor):
    
    _, d, h, w = tensor.size()
    
    tensor = tensor.view(d, h * w)
    
    gram = torch.mm(tensor, tensor.t()) / (d * h * w)
    
    return gram 

content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
style_grams = [gram_matrix(layer) for layer in style_features[:-1]]

target = content.clone().requires_grad_(True).to(device)

style_weights = {'conv1_1': 1.5,
                 'conv2_1': 0.80,
                 'conv3_1': 0.25,
                 'conv4_1': 0.25,
                 'conv5_1': 0.25}

content_weight = 1e-2  
style_weight = 1e9  

show = 100

optimizer = optim.Adam([target], lr=0.01)
steps = 5000  

show_tensor_image(un_normalize_image(target.reshape((3,150,225)), image_mean, image_std))

for i in range(1, steps+1):
    
    target_features = get_features(target, vgg)
    
    content_loss = torch.mean((target_features[-1] - content_features[-1])**2)
    
    style_loss = 0
    for layer in range(0, 5):
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        # layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        layer_style_loss = torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (d * h * w)
        
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if  i % show == 0:
        print(f"epoch: {i}, c_loss: {content_weight * content_loss}, s_loss: {style_weight * style_loss}, total: {total_loss}")

show_tensor_image(un_normalize_image(target.reshape((3,150,225)), image_mean, image_std))
