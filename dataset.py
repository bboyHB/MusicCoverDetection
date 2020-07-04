import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

"""
代码参考：https://cuijiahua.com/blog/2020/03/dl-16.html
"""


class TextSegDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # TODO
        # 1. Initialize file paths or a list of file names.
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'mask')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        # image = image / 255
        if label.max() > 1:
            label = label / 255
        # image = torch.tensor(image, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.float32)
        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgs_path)


