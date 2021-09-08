# This program is trial on algorithm about matching anchor boxes with target anchor boxes. 
# Does not contain training part
# WIP: being able to display anchor boxes but target detection unimplemented
import torch
from math import sqrt
from PIL import Image
import matplotlib.pyplot as plt
import sys

from torchvision import transforms
sys.path.append('../machine_learning/')
from utils.functions import apply_anchor_boxes

# get an experiment image for showing anchor box
img = Image.open("training_data/images/autumn_oak.jpg")
image = transforms.ToTensor()(img)
fig = plt.imshow(img)
# apply_anchor_boxes(fig, [[20,30, 500, 600]])
# plt.show()


# define ground truth
ground_truth = [[50, 30, 600, 800], [750, 30, 1400, 800]]
apply_anchor_boxes(fig, ground_truth, box_colour='black')

# define all gussed anchor boxes
print(f"shape = {image.shape}")
h, w = image.shape[1], image.shape[2]
gussed_centers = [[400, 400]]
gussed_boxes = []
sizes = [0.75, 0.5, 0.25]
ratios=[1, 2, 0.5]
# actual size ratio pairs used for defining anchor boxes
r_s_pairs = [[sizes[0], r] for r in ratios] + [[s, ratios[0]] for s in sizes]
# create anchor boxes
for center_x, center_y in gussed_centers:
  for s, r in r_s_pairs:
    height, width = h * s / sqrt(r), w * s * sqrt(r)
    gussed_boxes.append([center_x - width/2, 
                         center_y - height/2, 
                         center_x + width/2, 
                         center_y + height/2,])
apply_anchor_boxes(fig, gussed_boxes, box_colour='blue')

plt.show()