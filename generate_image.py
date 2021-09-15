import csv
import os
import random
import string

import pandas as pd
from PIL import Image, ImageFilter
from PIL.ImageDraw import Draw
from captcha.image import ImageCaptcha, table


class CustomImageCaptcha(ImageCaptcha):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def create_captcha_image(self, chars, color, background):
        """Create the CAPTCHA image itself.
        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.
        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            im = Image.new('RGBA', (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-10, 10), Image.BILINEAR, expand=1)

            # warp
            dx = w * random.uniform(0.1, 0.3)
            dy = h * random.uniform(0.2, 0.3)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1, y1,
                -x1, h2 - y2,
                w2 + x2, h2 + y2,
                w2 - x2, -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), Image.QUAD, data)
            return im

        images = []
        for c in chars:
            if random.random() > 0.2:
                images.append(_draw_character(" "))
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.1 * average)
        offset = int(average * 1.2)

        for im in images:
            w, h = im.size
            mask = im.convert('L').point(table)
            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            image = image.resize((self._width, self._height))

        return image

    def generate_image(self, chars):
        """Generate the image of the given characters.
        :param chars: text to be generated.
        """
        # background = random_color(238, 255)
        # color = random_color(10, 200, random.randint(220, 255))
        background = (255, 255, 255)
        color = (0, 255, 0, 255)
        im = self.create_captcha_image(chars, color, background)
        # self.create_noise_dots(im, color)
        self.create_noise_curve(im, color)
        self.create_noise_curve(im, color)
        # im = im.filter(ImageFilter.SMOOTH)
        return im


def text_generate(length=5):
    text = ''.join(
        random.choice(string.digits + string.ascii_lowercase)
        for _ in range(length))
    return text


image_dir = '/media/data_it/Data_set/database_image/ocr/captcha/zalo_generate'
n = 10000
height = 100
width = height * 4
length_text = 5
override = True

filenames = []
texts = []
image = CustomImageCaptcha(width=width, height=height, fonts=[
    "/media/thiennt/projects/cv_end_to_end/training/ocr/crnn_ctc/datasets/fonts/font_03.ttf",
],
                           font_sizes=None)
for i in range(n):
    filename = f'{i:0>5}.png'
    filepath = os.path.join(image_dir, filename)
    if os.path.exists(filepath) and not override:
        continue
    text = text_generate(length_text)
    image.write(text, filepath)
    print(f'Generate {i}: {text} : {filename}')

    filenames.append(filename)
    texts.append(text)

df = pd.DataFrame({
    "filename": filenames,
    "label": texts
})


df.to_csv(image_dir + ".csv", quoting=csv.QUOTE_ALL, index=False)