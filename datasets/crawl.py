import cv2
import numpy as np
# import pytesseract
import traceback
import requests
from PIL import Image
from io import BytesIO
import os

# import demo
# from paddleocr import PaddleOCR

url = "https://dichvucong.bocongan.gov.vn/captcha.php?act=create&t=1684377592487&for=maXacNhan"
img_dir = "/Users/tienthien/workspace/mine/tf-crnn-ctc/datasets/dichvucong/test"


def preprocess(img):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    return dilation


def read(img):
    img_ = preprocess(img)
    img_ = img
    try:
        text = pytesseract.image_to_string(img_, lang='eng', config="--oem 3 --psm 8")
        text = ''.join([c for c in text if c.isalnum()])
        return text
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return None


def pretrained_read(img):
    pred = demo.inference(img)
    return pred


# ocr = PaddleOCR(use_angle_cls=False, lang='en')  # need to run only once to download and load model into memory


def read_paddle_ocr(img):
    result = ocr.ocr(img, det=True, cls=False)
    for line in result:
        return line[1][0]


def download(n_samples):
    start_id = 0

    for i in range(start_id, n_samples):
        cookies = {
        }

        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7,fr-FR;q=0.6,fr;q=0.5',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
        }
        response = requests.get(url,
                                headers=headers, cookies=cookies,
                                verify=False)
        # print(response.status_code)
        # print(response.content)
        img = Image.open(BytesIO(response.content))
        img = np.array(img)[:, :, :3]
        img = img[:, :, ::-1]
        label = ""
        # label = pretrained_read(img)
        # label = read_paddle_ocr(img)
        # label = read(img)

        filename = "{:05}_{}.png".format(i, label)
        cv2.imwrite(os.path.join(img_dir, filename), img)
        # img.save(os.path.join(img_dir, filename))


if __name__ == '__main__':
    download(3000)
