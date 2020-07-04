from PIL import Image,ImageDraw, ImageFont
import random
import numpy
import webcolors
from scipy import misc
import matplotlib.pyplot as plt
import cv2


def judge(result):
    flag=1
    for r in result:
        if r!=True:
            flag=0
    return flag


#raw_pic_color为背景颜色，left为文字左上角那个点到图片左边框的距离，right，up，bottom分别为右，上，底
#text为文本，text_color为文字颜色，font为字体样式（类型为string，对应的样式名),angel为角度，最后一个是文件名

def gen_data(raw_pic_color,left,right,up,bottom,text,text_color,text_size,text_font,angel,file_name):
    img=Image.new("RGB",(10000,10000),raw_pic_color)
    font = ImageFont.truetype("fonts/"+text_font,text_size)
    draw = ImageDraw.Draw(img)
    height=5000
    width=5000
    draw.text((width,height),text, fill=text_color,font=font)
    img=img.rotate(angel)
    box=(width-left,height-up,width+right,height+bottom)
    img=img.crop(box)
    img.save("image/"+file_name)
    img=numpy.array(img)
    binary=numpy.zeros((up+bottom,right+left),numpy.uint8)
    for i in range(up+bottom):
        for j in range(left+right):
            if judge(webcolors.name_to_rgb(raw_pic_color)==img[i,j]):
                binary[i,j]=0
            else:
                binary[i,j]=255
    binary=Image.fromarray(binary,mode='L')
    binary.save("mask/"+file_name)
    



    
