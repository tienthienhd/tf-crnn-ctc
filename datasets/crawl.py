import cv2
import numpy as np
import requests
import urllib3
from PIL import Image
from io import BytesIO
import os
import ssl
from urllib3 import poolmanager

url = "https://digiapp.vietcombank.com.vn/utility-service/v1/captcha/1863f5fe-a821-7713-f2d1-3184737fdfb6"
img_dir = "/media/thiennt/data/backup_old_ubuntu/data_id/projects/mine/cv_end_to_end/training/ocr/crnn_ctc/datasets/vietcombank2/image"
pre_label = True;


# import demo
def pretrained_read(img):
    pred = demo.inference(img)
    return pred


if pre_label:
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=False, lang='en')  # need to run only once to download and load model into memory


def read_paddle_ocr(img):
    result = ocr.ocr(img, det=False, cls=False)
    for line in result:
        return line[0][0]


class CustomHttpAdapter(requests.adapters.HTTPAdapter):
    '''Transport adapter" that allows us to use custom ssl_context.'''

    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, ssl_context=self.ssl_context)


def get_downloaded():
    downloaded = set()
    for filename in os.listdir(img_dir):
        file_id, ext = os.path.splitext(filename)
        number, label = file_id.split('_')
        downloaded.add(int(number))
    return downloaded


def download(n_samples):
    start_id = 0
    downloaded_number = get_downloaded()

    for i in range(start_id, n_samples):
        if i in downloaded_number:
            continue

        cookies = {
        }

        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9,vi-VN;q=0.8,vi;q=0.7',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'Cookie': 'TS01388157=011fc56c76793a6f687ebdf12da79ca46e66fd578a151aa5fd59007944c82371b47dba254cc0939078553484522bbea6d7350a540c',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"'
        }
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ctx.options |= 0x4
        session = requests.session()
        session.mount('https://', CustomHttpAdapter(ctx))
        response = session.get(url, headers=headers, cookies=cookies)

        print(f'download image: {i} / {n_samples}: {response.status_code}')
        img = Image.open(BytesIO(response.content))
        img = np.array(img)[:, :, :3]
        img = img[:, :, ::-1]
        label = ""
        if pre_label:
            # label = pretrained_read(img)
            label = read_paddle_ocr(img)
            # label = read(img)

        filename = "{:05}_{}.png".format(i, label)
        cv2.imwrite(os.path.join(img_dir, filename), img)
        # img.save(os.path.join(img_dir, filename))


if __name__ == '__main__':
    download(3000)
