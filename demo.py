import argparse
import glob
import os

import cv2
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import string
import numpy as np
from model import CTCLayer
import config



def inference(img):
    # img = np.expand_dims(img, axis=-1)
    x = np.expand_dims(img, axis=0)

    y_pred = model.predict(x)
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:,
          :config.DatasetConfig.max_len]
    out = ''.join([characters[x] if x >= 0 else '' for x in out[0]])
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dichvucong')
    parser.add_argument('--cfg', type=str, default="config.json")

    args = parser.parse_args()

    config.load_config(f'datasets/{args.data}/models/config.json')

    model_path = f"datasets/{args.data}/models/last_inference_model.h5"
    img_dir = list(sorted(glob.glob(config.DatasetConfig.image_dir + '/0299*'), reverse=True))

    model = load_model(model_path)

    characters = config.DatasetConfig.charset

    if args.cfg != 'config.json':
        cfg = args.cfg
    else:
        cfg = os.path.join(f'./datasets/{args.data}/models/config.json')

    config.load_config(cfg)
    from PIL import Image
    for file in img_dir:
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        h, w, c = img.shape
        new_h = config.DatasetConfig.height
        w = int(w * new_h / h)
        img = cv2.resize(img, (w, new_h))
        if config.DatasetConfig.depth == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)

        img_ = img / 255.0
        pred = inference(img_)

        print(f'{file}: {pred}')
        cv2.imshow(pred, img)
        cv2.waitKey(0)
