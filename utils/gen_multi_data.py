from PIL import Image, ImageDraw, ImageFont
import random
import numpy
import os


def judge(result):
    flag = 1
    for r in result:
        if r != True:
            flag = 0
    return flag


# raw_pic_color为背景颜色，left为文字左上角那个点到图片左边框的距离，right，up，bottom分别为右，上，底
# text为文本，text_color为文字颜色，font为字体样式（类型为string，对应的样式名),angel为角度，最后一个是文件名

def gen_data(raw_pic_color,
             img_num,
             width,
             height,
             file_name,
             gen_path):
    string=[]
    img = Image.new("RGB", (width, height), raw_pic_color)
    for i in range(img_num):
        font = ImageFont.truetype(os.path.join("fonts/", random_fonts()),random.randint(20,70))
        draw = ImageDraw.Draw(img)
        text=random_sentence()
        length=len(text)
        flag=1
        h=0
        w=0
        while(1):
            w=random.randint(0,width-1)
            h=random.randint(0,height-1)
            flag=1
            for str in string:
                if w-str[0]>=0 and w-str[0]<=str[3] and h-str[1]>=0 and h-str[1]<=50:
                    flag=0
                    break
            if flag==1 and w+length*50<=width and h+50<=height:
                string.append([w,h,50*length,50])
                break
        draw.text((w,h), text, fill=random_RGB_color(), font=font)
        #img = img.rotate(angel)
        #box = (width - left, height - up, width + right, height + bottom)
        #img = img.crop(box)
    img.save(os.path.join(gen_path, "multi_image/", file_name))
    img = numpy.array(img)
    binary = numpy.zeros((height, width,3), numpy.uint8)
    for i in range(height):
        for j in range(width):
            if all(raw_pic_color == img[i, j]):
                binary[i, j] = [0,0,0]
            else:
                binary[i, j] = img[i,j]
    binary = Image.fromarray(binary)
    binary.save(os.path.join(gen_path, "multi_mask/", file_name))


def random_RGB_color():
    return (random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255))


def random_sentence(size=10):
    alphabet = 'abcdefghijklmnopqrstuvwxyz' + 'abcdefghijklmnopqrstuvwxyz'.upper() + ',.;:'
    sentence = ''
    for i in range(random.randint(1, size)):
        sentence += random.choice(alphabet)
    return sentence


def random_fonts():
    all_font = os.listdir("fonts/")
    ran_font = all_font[random.randint(0, len(all_font) - 1)]
    return ran_font


def random_lrud(length):
    return (random.randint(10, 500),
            random.randint(length*50, 550),
            random.randint(10, 500),
            random.randint(50, 500))


if __name__ == '__main__':
    data_dir = '../sythdata'
    image_dir = os.path.join(data_dir, 'multi_image')
    mask_dir = os.path.join(data_dir, 'multi_mask')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    for i in range(1):
        gen_data(random_RGB_color(),
                 random.randint(1,5),
                 random.randint(480,960),
                 random.randint(480,960),
                 f'{i}.jpg',
                 data_dir)