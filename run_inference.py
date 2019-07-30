#%%
from __future__ import absolute_import, division, print_function
from torch.autograd import Variable
from torch.utils import data
import torch
import torch.hub
import torch.nn.functional as F
import torch
import json
import multiprocessing
import os
import click
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from PIL import Image
from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy.special import expit
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
#%%
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
if cuda:
    print("Device:")
    for i in range(torch.cuda.device_count()):
        print("    {}:".format(i), torch.cuda.get_device_name(i))
else:
    print("Device: CPU")

#%%
ITER_MAX= 10
POS_W= 3
POS_XY_STD= 1
BI_W= 4
BI_XY_STD= 67
BI_RGB_STD= 3
n_classes = 182
def crf(image, probmap,img_w,img_h):
    w = img_w
    h = img_h
    print('prob shape')
    print(probmap.shape,probmap.size)
    C, H, W = probmap.shape
    # probmap = probmap.transpose(2, 0, 1).copy(order='C')
    d = dcrf.DenseCRF2D(W, H, n_classes)
    # U = -np.log(probmap) # Unary potential.
    U = utils.unary_from_softmax(probmap)
    U = U.reshape((n_classes, -1)) # Needs to be flat.
    d.setUnaryEnergy(U)
    image = np.ascontiguousarray(image)
    image = image.reshape((H,W,3))
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=BI_XY_STD, srgb=BI_RGB_STD,\
         rgbim=image, compat=BI_W)

    Q = d.inference(ITER_MAX)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, H, W))#.transpose(1, 2, 0)
  

    return preds

#%%
model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", n_classes=182)
model.load_state_dict(torch.load("./deeplabv2_resnet101_msc-cocostuff164k-100000.pth"))
model.eval()
#%%
file_name = 'test1'
filename = './'+file_name+'.jpg'
image = cv2.imread(filename)
image_og = image
#%%

# Subtract mean values
image = image.astype(np.float32)
image -= np.array(
    [
        float(104.008),
        float(116.669),
        float(122.675),
    ]
)
img_arr = image.transpose(2, 0, 1) # C x H x W
img_arr = np.expand_dims(img_arr,axis = 0)


img_tensor = torch.from_numpy(img_arr)
img_tensor = img_tensor.type('torch.FloatTensor')

# sanity check
# print(img_arr.shape)
# print(img_tensor.shape,img_tensor.size)
# print(dim_size)

#%%
x = model(img_tensor)
print(x[0].shape,x[0].size)
og_x = x

print(type(x))
logit = x
#%%
logit = logit.detach()
logit = logit.cpu().numpy()

print(logit.shape,logit.size)
print(logit[0].shape,logit[0].size)
#%%
_,_, H, W = img_tensor.shape
logit = torch.FloatTensor(logit)
logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
prob = F.softmax(logit, dim=1)[0].numpy()

image = image.astype(np.uint8).transpose(1, 2, 0)
#%%
prob = crf(image, prob,dim_size,dim_size)
labelmap = np.argmax(prob, axis=0)

#%%
# sanity check
# print('after softmax')
# print(prob.shape,prob.size,type(prob))
# print('*********************************************')
# print('')
# print('after argmax')
# print(labelmap.shape,labelmap.size,type(labelmap))

#%%
unique_labels, index = np.unique(labelmap,return_counts=True)
print(unique_labels)
colormap = 'jet'
cm = plt.get_cmap(colormap)
imp3 = (cm(labelmap)[:, :, :3] * 255).astype(np.uint8)
print(imp3.shape,imp3.size)
plt.imshow(imp3)
#%%

#%%
blended = cv2.addWeighted(image_og, 0.5, imp3, 0.7,0)
plt.imshow(blended)
#%%

labels = np.unique(labelmap)
stuff_mask_list = []
objects_mask_list = []

#get labels from labels.txt
classes = {}
labels_filename = './labels.txt'
with open(labels_filename) as f:
       
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]
for i in labels:
    print(classes[i])

# Show result for each class
rows = np.floor(np.sqrt(len(labels) + 1))
cols = np.ceil((len(labels) + 1) / rows)

print(image_og.shape,image_og.size)

plt.figure(figsize=(10, 10))
ax = plt.subplot(rows, cols, 1)
ax.set_title("Input image")
ax.imshow(raw_image[:, :, ::-1])
ax.axis("off")
image_list = []
for i, label in enumerate(labels):
    mask = labelmap == label
    ax = plt.subplot(rows, cols, i + 2)
  
    ax.set_title(classes[label])
       
    ax.imshow(raw_image[..., ::-1])
    tmp = mask.astype(np.float32)
    if label>0 and label<=91:
        objects_mask_list.append((tmp,classes[label]))
        ax.imshow(mask.astype(np.float32), alpha=0.5)
    if label>91 and label<=182:
        stuff_mask_list.append((tmp,classes[label]))
        ax.imshow(mask.astype(np.float32), alpha=0.5)

    ax.imshow(mask.astype(np.float32), alpha=0.5)
    ax.axis("off")
ax = plt.subplot(rows,cols,i+2)
ax.set_title('blended')
ax.imshow(blended)
plt.tight_layout()
save_name = './'+file_name+'.pdf'
plt.savefig(save_name)
plt.close()
#%%
l1 = len(stuff_mask_list)
l2 = len(objects_mask_list)
print(l1,l2)

#%%
contour_area = 0
contour_area_max = img_area/10

contour_objects_list = []

contour_stuff_list = []

sorted_contour_objects_list = []
sorted_contour_stuff_list = []

def get_contour_info(mask_list):
    contour_things_area_list = []
    contour_things_list = []
    for things in mask_list:
        things_img_unconverted = things[0]
        things_img = np.uint8(things_img_unconverted)
        things_things, hierarchy = cv2.findContours(things_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for ts in things_things:
            if (cv2.contourArea(ts)):
                contour_area = cv2.contourArea(ts)
                contour_things_area_list.append((contour_area,things[1]))
                contour_things_list.append((ts,things[1]))
    return contour_things_area_list,contour_things_list

contour_objects_area_list,contour_objects_list = get_contour_info(objects_mask_list)

contour_stuff_area_list,contour_stuff_list = get_contour_info(stuff_mask_list)


#%%
final_stuff_list = []
final_objects_list = []
sorted_contour_stuff_area = []
sorted_contour_objects_area = []

sorted_contour_stuff_area = sorted(contour_stuff_area_list,key = lambda x: x[0],reverse = True)
sorted_contour_objects_area = sorted(contour_objects_area_list,key = lambda x: x[0],reverse = True)


def final_list(sorted_list):
    final_list = []
    count = 0
    for i,j in sorted_list:
        final_list.append(j)
        # print(i,j)
        count = count +1
        if count == 5:
            break
    return final_list

final_stuff_list = final_list(sorted_contour_stuff_area)
final_objects_list = final_list(sorted_contour_objects_area)
#%%
def drop_duplicates(things_list):
    return list(dict.fromkeys(things_list))
# print('before sorting desc')
# print(final_objects_list)
# print()
# print(final_stuff_list)
# print()
final_stuff_list = drop_duplicates(final_stuff_list)
final_objects_list = drop_duplicates(final_objects_list)
# print(final_objects_list)
# print()
# print(final_stuff_list)
# print()


#%%
w = image_og.shape[0]
h = image_og.shape[1]
z = image_og.shape[2]
# print(w,h,z)
img = np.zeros([w, h], dtype=np.float32)
print(img.shape[0],img.shape[1])

def final_mask_maker(things_list,final_things_list,zero_img):
    img = zero_img
    for mask,label in things_list:
        if label in final_things_list:
            img = cv2.add(img,mask)
            # cv2.imshow(str(label),mask)
            # cv2.waitKey()
    img = (img * 255).round().astype(np.uint8)
    return img

final_stuff_mask = final_mask_maker(stuff_mask_list,final_stuff_list,img)
final_objects_mask = final_mask_maker(objects_mask_list,final_objects_list,img)

# print(final_stuff_mask)
ret,thresh1 = cv2.threshold(final_stuff_mask,127,255,cv2.THRESH_BINARY)
image_type = 'composite mask'
filename = './'+file_name+'.jpg'
f_string = './'+file_name+' '+image_type+'.jpg'

cv2.imshow(image_type,thresh1)
cv2.imwrite(f_string,thresh1)
cv2.waitKey()
cv2.destroyAllWindows()
#%%
contours, hierarchy = cv2.findContours(final_stuff_mask,\
    cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cimg = cv2.drawContours(final_stuff_mask, contours, -1, (255,255,255), 8)
image_type = 'edge mask'
filename = './'+file_name+'.jpg'
f_string = './'+file_name+' '+image_type+'.jpg'

edges = cv2.Canny(cimg,100,150)
cv2.imshow(image_type,edges)
cv2.imwrite(f_string,edges)
cv2.waitKey()
cv2.destroyAllWindows()
