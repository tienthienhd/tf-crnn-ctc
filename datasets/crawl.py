import cv2
import numpy as np
import pytesseract
import traceback
import requests
from PIL import Image
from io import BytesIO
import os
import demo

url = "http://ag.sgd168.com/VerifyCode.aspx?AspxAutoDetectCookieSupport=1"
img_dir = "/media/data_it/thiennt/cv_end_to_end/training/ocr/crnn_ctc/datasets/ag_sgd168/test"


def preprocess(img):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    return dilation


def read(img):
    # img_ = preprocess(img)
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


def download(n_samples):
    start_id = 0
    for i in range(start_id, n_samples):
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:79.0) Gecko/20100101 Firefox/79.0", "Cookie": "Cookie: AspxAutoDetectCookieSupport=1; ASP.NET_SessionId=hmdzv145kr2zlcz4nlkhck45; No.VerifyCode=CcrRKz0St9eGBkgbSx/faQ=="}, verify=False)
        img = Image.open(BytesIO(response.content))
        img = np.array(img)[:, :, :3]
        label = pretrained_read(img)
        filename = "{:05}_{}.png".format(i, label)
        cv2.imwrite(os.path.join(img_dir, filename), img)


if __name__ == '__main__':
    download(200)
