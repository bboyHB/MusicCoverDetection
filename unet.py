import torch
import torchvision
from torch import nn


# input_size = 572
# size1 = input_size - 2 - 2
# size2 = size1 // 2 - 2 - 2
# size3 = size2 // 2 - 2 - 2
# size4 = size3 // 2 - 2 - 2
# size5 = size4 // 2 - 2 - 2
# size6 = size5 * 2
# size7 = (size6 - 2 - 2) * 2
# size8 = (size7 - 2 - 2) * 2
# size9 = (size8 - 2 - 2) * 2
# size10 = size9 - 2 - 2



class UNet(nn.Module):
    """
    U-Net 结构实现 照着论文搭积木 论文地址：https://arxiv.org/pdf/1505.04597.pdf

    一些函数和变量的名称参考图片 ‘./U-Net.png’

    部分内容参考这个博客：https://cuijiahua.com/blog/2019/12/dl-15.html
    """
    def __init__(self):
        super(UNet, self).__init__()
        in_channels = 1
        out_channels = 1
        basic_channels = 64
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
        img = nn.functional.pad(img, [4, 4,
                                      4, 4])
        temp1 = self.down1(img)
        temp2 = self.down2(self.pool1(temp1))
        temp3 = self.down3(self.pool2(temp2))
        temp4 = self.down4(self.pool3(temp3))
        img = self.upconv1(self.plain(self.pool4(temp4)))
        img = self.pad_concat(temp4, img)
        img = self.upconv2(self.up1(img))
        img = self.pad_concat(temp3, img)
        img = self.upconv3(self.up2(img))
        img = self.pad_concat(temp2, img)
        img = self.upconv4(self.up3(img))
        img = self.pad_concat(temp1, img)
        img = self.up4(img)
        return img

    def pad_concat(self, temp, img):
        """
        论文中虽然写的是crop，但是用pad会更加方便，而且保留更多信息。

        ‘官方的写法，用的pad，所以我这里也用pad了。
        其实这个不用太纠结，unet的思想主要在于这种U型的经典结构。
        至于pad或者crop其实，可以根据需求进行调整的。’   --引用自Jack Cui（上文中博客的博主）

        """
        deltaY = temp.shape[2] - img.shape[2]
        deltaX = temp.shape[3] - img.shape[3]
        img = nn.functional.pad(img, [deltaX // 2, deltaX - deltaX // 2,
                                      deltaY // 2, deltaY - deltaY // 2])
        img = torch.cat((temp, img), dim=1)
        return img

if __name__ == '__main__':
    u = UNet()
    a = torch.rand((1, 3, 600, 600))
    a = u(a)
    print(a.shape)
