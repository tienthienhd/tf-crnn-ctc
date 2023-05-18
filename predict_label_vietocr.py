import os
import ssl

import pandas as pd
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

ssl._create_default_https_context = ssl._create_unverified_context

config = Cfg.load_config_from_name('vgg_transformer')

# config['weights'] = './weights/transformerocr.pth'
# config['cnn']['pretrained']=False
config['device'] = 'cpu'

detector = Predictor(config)

from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory



vietocr = False
data_dir = '/Users/tienthien/workspace/mine/tf-crnn-ctc/datasets/dichvucong/train'
label_file = '/Users/tienthien/workspace/mine/tf-crnn-ctc/datasets/dichvucong/label2.csv'
flag_header = not os.path.exists(label_file)
processed = set()
if not flag_header:
    df = pd.read_csv(label_file, dtype={"filename": str, "label": str})
    for idx, row in df.iterrows():
        processed.add(row['filename'])
with open(label_file, 'a') as f:
    if flag_header:
        f.write('filename,label,conf\n')
    for filename in sorted(os.listdir(data_dir)):
        if filename in processed:
            continue
        image_path = os.path.join(data_dir, filename)
        if vietocr:
            img = Image.open(image_path)
            label, conf = detector.predict(img, return_prob=True)
        else:
            result = ocr.ocr(image_path, cls=True)
            try:
                label, conf = result[0][0][1]
            except:
                print("Error: ", result)
                label, conf = None, None

        f.write(f'"{filename}","{label}",{conf}\n')
