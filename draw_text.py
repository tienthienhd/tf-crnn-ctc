import os.path
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_text(text, width, height, font, font_size=1, text_color=(0, 0, 0), output_filepath='test.jpg'):
    font = ImageFont.truetype("datasets/fonts/Walkway Black RevOblique.ttf", 40)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((12, 6), text, (113, 193, 216), font=font)
    draw = ImageDraw.Draw(img)
    img.save(output_filepath)
    # img.show('test')


def gen_text(length, charset):
    text = ''.join(
        random.choice(charset)
        for _ in range(length))
    return text


if __name__ == '__main__':
    num = 10000
    length = 5
    charset = 'abcdefghijklmnopqrstuvwxyz'
    output_dir = '/Users/tienthien/workspace/mine/tf-crnn-ctc/datasets/dichvucong/generate'
    label_file = '/Users/tienthien/workspace/mine/tf-crnn-ctc/datasets/dichvucong/generate.csv'
    with open(label_file, 'w') as f:
        f.write("filename,label\n")
        for i in range(num):
            print("generate ", i)
            text = gen_text(length, charset)
            filename = "{:05}.png".format(i)
            output_path = os.path.join(output_dir, filename)
            draw_text(text,  165, 50, None, output_filepath=output_path)
            f.write(f'"{filename}","{text}"\n')



