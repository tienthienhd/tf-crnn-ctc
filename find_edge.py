import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot():
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(closing, cmap='gray')
    plt.title('Closing Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def process(img):
    edges = cv2.Canny(img, 100, 200)
    return edges


def run_process(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        img = cv2.imread(filepath)
        output_img = process(img)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, output_img)


if __name__ == '__main__':
    run_process('/media/data_it/Data_set/database_image/ocr/captcha/garena',
                '/media/data_it/Data_set/database_image/ocr/captcha/garena_processed')
