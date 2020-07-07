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
             lrub,
             text,
             text_color,
             text_size,
             text_font,
             angel,
             file_name,
             gen_path):
    left, right, up, bottom = lrub
    img = Image.new("RGB", (10000, 10000), raw_pic_color)
    font = ImageFont.truetype(os.path.join("fonts/", text_font), text_size)
    draw = ImageDraw.Draw(img)
    height = 5000
    width = 5000
    draw.text((width, height), text, fill=text_color, font=font)
    img = img.rotate(angel)
    box = (width - left, height - up, width + right, height + bottom)
    img = img.crop(box)
    img.save(os.path.join(gen_path, "image/", file_name))
    img = numpy.array(img)
    binary = numpy.zeros((up + bottom, right + left), numpy.uint8)
    for i in range(up + bottom):
        for j in range(left + right):
            if all(raw_pic_color == img[i, j]):
                binary[i, j] = 0
            else:
                binary[i, j] = 255
    binary = Image.fromarray(binary, mode='L')
    binary.save(os.path.join(gen_path, "mask/", file_name))


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


def random_lrud():
    return (random.randint(10, 500),
            random.randint(10, 500),
            random.randint(10, 500),
            random.randint(10, 500))


if __name__ == '__main__':
    data_dir = '../sythdata'
    image_dir = os.path.join(data_dir, 'image')
    mask_dir = os.path.join(data_dir, 'mask')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    for i in range(200):
        gen_data(random_RGB_color(),
                 random_lrud(),
                 random_sentence(),
                 random_RGB_color(),
                 random.randint(10, 80),
                 random_fonts(),
                 0,
                 f'{i}.jpg',
                 data_dir)
