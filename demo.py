import glob

import cv2
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import string
import numpy as np

import config

model_path = "datasets/vr_plate/checkpoint/inference_model.h5"
img_dir = ['/media/data_it/Data_set/database_image/card/vr/info/train/plate_new/3534_59F1-229.01.png',
           '/media/data_it/Data_set/database_image/card/vr/info/train/plate_new/3535_60F1-5857.png',
           '/media/data_it/Data_set/database_image/card/vr/info/train/plate_new/3536_59V1-173.33.png',
           '/media/data_it/Data_set/database_image/card/vr/info/train/plate_new/3537_29Y3-376.53.png',
           '/media/data_it/Data_set/database_image/card/vr/info/train/plate_new/3538_29V7-374.72.png',
           '/media/data_it/Data_set/database_image/card/vr/info/train/plate_new/3539_99K1-0841.png']
model = load_model(model_path)

characters = config.dataset['charset']


def inference(img):
    # img = np.expand_dims(img, axis=-1)
    x = np.expand_dims(img, axis=0)

    y_pred = model.predict(x)
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :config.dataset['max_len']]
    out = ''.join([characters[x] for x in out[0]])
    return out


if __name__ == '__main__':
    for file in img_dir:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        h, w, c = img.shape
        new_h = config.dataset['height']
        w = int(w * new_h / h)
        img = cv2.resize(img, (w, new_h))
        img_ = img / 255.0 - 0.5
        pred = inference(img_)
        cv2.imshow(pred, img)
        cv2.waitKey(0)
