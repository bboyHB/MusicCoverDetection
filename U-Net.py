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
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.down2 = nn.Sequential(
            nn.Conv2d(basic_channels, basic_channels * 2, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 2, basic_channels * 2, 3),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.down3 = nn.Sequential(
            nn.Conv2d(basic_channels * 2, basic_channels * 4, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 4, basic_channels * 4, 3),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.down4 = nn.Sequential(
            nn.Conv2d(basic_channels * 4, basic_channels * 8, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 8, basic_channels * 8, 3),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.plain = nn.Sequential(
            nn.Conv2d(basic_channels * 8, basic_channels * 16, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 16, basic_channels * 16, 3),
            nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(basic_channels * 16, basic_channels * 8, 2, 2)
        self.up1 = nn.Sequential(
            nn.Conv2d(basic_channels * 16, basic_channels * 8, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 8, basic_channels * 8, 3),
            nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(basic_channels * 8, basic_channels * 4, 2, 2)
        self.up2 = nn.Sequential(
            nn.Conv2d(basic_channels * 8, basic_channels * 4, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 4, basic_channels * 4, 3),
            nn.ReLU()
        )
        self.upconv3 = nn.ConvTranspose2d(basic_channels * 4, basic_channels * 2, 2, 2)
        self.up3 = nn.Sequential(
            nn.Conv2d(basic_channels * 4, basic_channels * 2, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels * 2, basic_channels * 2, 3),
            nn.ReLU()
        )
        self.upconv4 = nn.ConvTranspose2d(basic_channels * 2, basic_channels, 2, 2)
        self.up4 = nn.Sequential(
            nn.Conv2d(basic_channels * 2, basic_channels, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels, basic_channels, 3),
            nn.ReLU(),
            nn.Conv2d(basic_channels, out_channels, 1)
        )

    def forward(self, img):
        temp1 = self.down1(img)
        temp2 = self.down2(self.pool1(temp1))
        temp3 = self.down3(self.pool2(temp2))
        temp4 = self.down4(self.pool3(temp3))
        img = self.upconv1(self.plain(self.pool4(temp4)))
        deltaY = temp4.shape[2] - img.shape[2]
        deltaX = temp4.shape[3] - img.shape[3]
        img = nn.functional.pad(img, [deltaX//2, deltaX - deltaX//2,
                                      deltaY//2, deltaY - deltaY//2])
        img = torch.cat((temp4, img), dim=1)
        img = self.upconv2(self.up1(img))
        deltaY = temp3.shape[2] - img.shape[2]
        deltaX = temp3.shape[3] - img.shape[3]
        img = nn.functional.pad(img, [deltaX // 2, deltaX - deltaX // 2,
                                      deltaY // 2, deltaY - deltaY // 2])
        img = torch.cat((temp3, img), dim=1)
        img = self.upconv3(self.up2(img))
        deltaY = temp2.shape[2] - img.shape[2]
        deltaX = temp2.shape[3] - img.shape[3]
        img = nn.functional.pad(img, [deltaX // 2, deltaX - deltaX // 2,
                                      deltaY // 2, deltaY - deltaY // 2])
        img = torch.cat((temp2, img), dim=1)
        img = self.upconv4(self.up3(img))
        deltaY = temp1.shape[2] - img.shape[2]
        deltaX = temp1.shape[3] - img.shape[3]
        img = nn.functional.pad(img, [deltaX // 2, deltaX - deltaX // 2,
                                      deltaY // 2, deltaY - deltaY // 2])
        img = torch.cat((temp1, img), dim=1)
        img = self.up4(img)
        return img

u = UNet()
a = torch.rand((1, 1, 572, 572))
a = u(a)
print(a.shape)
