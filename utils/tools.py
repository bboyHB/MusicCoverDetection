from PIL import Image
import os
from matplotlib import pyplot as plt

original_data_path = '../coverdata'
weight_hight_file_path = '../w_h.txt'

def width_height_extraction():
    """
    统计出所有数据集图片的宽和高，存储在文本文件中，方便后续分析使用
    :return:
    """
    with open(weight_hight_file_path, 'w') as wh:
        for p in os.listdir(original_data_path):
            temp_path = os.path.join(original_data_path, p)
            img = Image.open(temp_path)
            wh.write(str(img.size[0]) + ',' + str(img.size[1]) + '\n')


def plot_ratios():
    """
    统计数据集中每张图片的宽高信息及比例，并画图显示
    :return:
    """
    width_height = []
    with open(weight_hight_file_path) as wh:
        lines = wh.readlines()
        for line in lines:
            img_size = line.strip().split(',')
            width_height.append([float(img_size[0]), float(img_size[1])])
        ratios = [wh[1] / wh[0] for wh in width_height]
        areas = [wh[1] * wh[0] for wh in width_height]
        widths = [wh[0] for wh in width_height]
        heights = [wh[1] for wh in width_height]
        print('最大高：', max(heights),
              '最小高：', min(heights),
              '平均高：', sum(heights) / len(heights))
        print('最大宽：', max(widths),
              '最小宽：', min(widths),
              '平均宽：', sum(widths) / len(widths))
        print('最大高宽比：', max(ratios),
              '最小高宽比：', min(ratios),
              '平均高宽比：', sum(ratios) / len(ratios))
        print('最大面积：', max(areas),
              '最小面积：', min(areas),
              '平均面积：', sum(areas) / len(areas))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot([i for i in range(len(heights))], sorted(heights))
        plt.xlabel('第x个样本')
        plt.ylabel('高')
        plt.show()
        plt.plot([i for i in range(len(widths))], sorted(widths))
        plt.xlabel('第x个样本')
        plt.ylabel('宽')
        plt.show()
        plt.plot([i for i in range(len(ratios))], sorted(ratios))
        plt.xlabel('第x个样本')
        plt.ylabel('高宽比')
        plt.show()
        plt.plot([i for i in range(len(areas))], sorted(areas))
        plt.xlabel('第x个样本')
        plt.ylabel('面积')
        plt.show()
        plt.scatter([wh[0] for wh in width_height], [wh[1] for wh in width_height])
        plt.xlabel('宽')
        plt.ylabel('高')
        plt.show()


if __name__ == '__main__':
    width_height_extraction()
    plot_ratios()