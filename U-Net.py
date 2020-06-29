import torch
import torchvision
from torch import nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        in_channels = 1
        out_channels = 2
        basic_channels = 64
        input_size = 572
        size1 = input_size - 2 - 2
        size2 = size1 // 2 - 2 - 2
        size3 = size2 // 2 - 2 - 2
        size4 = size3 // 2 - 2 - 2
        size5 = size4 // 2 - 2 - 2
        size6 = size5 * 2
        size7 = (size6 - 2 - 2) * 2
        size8 = (size7 - 2 - 2) * 2
        size9 = (size8 - 2 - 2) * 2
        size10 = size9 - 2 - 2
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, basic_channels, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels, basic_channels, 3),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(
            nn.Conv2d(basic_channels, basic_channels * 2, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 2, basic_channels * 2, 3),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = nn.Sequential(
            nn.Conv2d(basic_channels * 2, basic_channels * 4, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 4, basic_channels * 4, 3),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = nn.Sequential(
            nn.Conv2d(basic_channels * 4, basic_channels * 8, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 8, basic_channels * 8, 3),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2)
        self.plain = nn.Sequential(
            nn.Conv2d(basic_channels * 8, basic_channels * 16, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 16, basic_channels * 16, 3),
            nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(basic_channels * 16, basic_channels * 8, 2)
        self.up1 = nn.Sequential(
            nn.Conv2d(basic_channels * 16, basic_channels * 8, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 8, basic_channels * 8, 3),
            nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(basic_channels * 8, basic_channels * 4, 2)
        self.up2 = nn.Sequential(
            nn.Conv2d(basic_channels * 8, basic_channels * 4, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 4, basic_channels * 4, 3),
            nn.ReLU()
        )
        self.upconv3 = nn.ConvTranspose2d(basic_channels * 4, basic_channels * 2, 2)
        self.up3 = nn.Sequential(
            nn.Conv2d(basic_channels * 4, basic_channels * 2, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 2, basic_channels * 2, 3),
            nn.ReLU()
        )
        self.upconv4 = nn.ConvTranspose2d(basic_channels * 2, basic_channels, 2)
        self.up4 = nn.Sequential(
            nn.Conv2d(basic_channels * 2, basic_channels, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels, basic_channels, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels, out_channels, 1)
        )
        self.center_crop1 = torchvision.transforms.CenterCrop(size6)
        self.center_crop2 = torchvision.transforms.CenterCrop(size7)
        self.center_crop3 = torchvision.transforms.CenterCrop(size8)
        self.center_crop4 = torchvision.transforms.CenterCrop(size9)

    def forward(self, img):
        temp1 = self.down1(img)
        temp2 = self.down2(self.pool1(temp1))
        temp3 = self.down3(self.pool2(temp2))
        temp4 = self.down4(self.pool3(temp3))
        img = self.upconv1(self.plain(self.pool4(temp4)))
        img = torch.cat((self.center_crop1(temp4), img), dim=0)
        img = self.upconv2(self.up1(img))
        img = torch.cat((self.center_crop2(temp3), img), dim=0)
        img = self.upconv3(self.up2(img))
        img = torch.cat((self.center_crop3(temp2), img), dim=0)
        img = self.upconv4(self.up3(img))
        img = torch.cat((self.center_crop4(temp1), img), dim=0)
        img = self.up4(img)
        return img


