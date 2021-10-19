# -*- coding = utf-8 -*-
import csv
import glob

from PIL import Image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image, preprocess_image
from torch import nn
from torchvision.models import resnet18
import pytorch_grad_cam
import cv2
import numpy as np
import os
import torch
from 模块化神经网络.net_Resnet18 import ResNet18
from torchvision import transforms


savepath = r'C:\Python\pycharm\...'

'''1.加载模型'''
# model = ResNet18(2)
# posttrain_model = model.load_state_dict(torch.load(savepath + r'\bestmodel.mdl'))
# print('------ok!--------')


trained_model = resnet18(pretrained=True)
full_connect_num=trained_model.fc.in_features
trained_model.fc=nn.Linear(full_connect_num,2)
model=trained_model
posttrain_model = model.load_state_dict(torch.load(savepath + r'\lastmodel.mdl'))
print('------ok!--------')

# print(model)
# target_layer = model.blk4.conv2
# print(target_layer)

target_layer = model.layer4[1].conv2
# print(target_layer)


def load_csv(root, filename):
    if not os.path.exists(os.path.join(root, filename)):
        images_path_list = []
        for name in classifylabel.keys():
            images_path_list += glob.glob(os.path.join(root, name, '*.png'))
            images_path_list += glob.glob(os.path.join(root, name, '*.jpg'))
            images_path_list += glob.glob(os.path.join(root, name, '*.jpeg'))
        with open(os.path.join(root, filename), mode='w', newline='') as f:  # csv
            writer = csv.writer(f)
            for imgpath in images_path_list:  # '.\\pokemon\\bulbasaur\\00000000.png'
                name = imgpath.split(os.sep)[-2]  # 将['pokemon','bulbasaur','00000000.png'],倒数第二个是文字标签
                label = classifylabel[name]  # 利
                writer.writerow([imgpath, label])  # '.\\mask_dataset\\image_nomask\\0650.jpg, 0'这种形式逐行写入filename.csv文件
            print('writen into csv file:', filename)
    else:
        # read from csv file
        imagespath, labels = [], []
        with open(os.path.join(root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # '.\\mask_dataset\\image_nomask\\0650.jpg', 0
                imgpath, strlabel = row
                label = int(strlabel)
                imagespath.append(imgpath)
                labels.append(label)
    return imagespath, labels
root = r"C:\Users\..."
classifylabel = {}
for name in sorted(os.listdir(root)):
    if not os.path.isdir(os.path.join(root, name)):
        continue
    else:
        classifylabel[name] = len(classifylabel.keys())
imagespath_list, _ = load_csv(root,'images.csv')
for i in np.arange(len(imagespath_list)):
    image_path = imagespath_list[i]
    label = image_path.split(os.sep)[-2]
    pictrue_name = image_path.split(os.sep)[-1]
    resize=224
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img_resize  = cv2.resize(rgb_img,(224,224),interpolation=cv2.INTER_AREA)  #将原始图片resize一样的大小
    imgdata_from_path = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((int(resize), int(resize))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = imgdata_from_path(image_path)
    input_tensor = img_tensor.unsqueeze(0)
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=False)
    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = None # 281
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb_img_resize, grayscale_cam)  # (224, 224, 3)
    cv2.imwrite(r'C:\Users\...'+ '\\' + label + '_' + pictrue_name, visualization)
    print(str(i+1))