import argparse
import csv
import glob
import os

import cv2
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import string
import numpy as np

import config


def inference(model, img, characters):
    # img = np.expand_dims(img, axis=-1)
    x = np.expand_dims(img, axis=0)

    y_pred = model.predict(x)
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:,
          :config.DatasetConfig.max_len]
    out = ''.join([characters[x] if x >= 0 else '' for x in out[0]])
    return out


def get_label(file_path, model, characters):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (120, 50))
    img = np.expand_dims(img, axis=-1)
    img_ = img / 255.0
    pred = inference(model, img_, characters)
    return pred


def run(data_name, image_dir, label_file, output_file, start, end):
    model_path = f"datasets/{data_name}/models/{data_name}.h5"
    model = load_model(model_path)
    characters = config.DatasetConfig.charset

    df_label = pd.read_csv(label_file, dtype={'label': str})

    df_label['filepath'] = df_label['filename'].apply(lambda x: os.path.join(image_dir, x))
    df_label['label'].iloc[start: end] = df_label['filepath'].iloc[start: end].apply(
        lambda x: get_label(x, model, characters))
    df_label.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='evisa')
    parser.add_argument('--cfg', type=str, default="evisa.json")

    args = parser.parse_args()

    if args.cfg != 'vietcombank2.json':
        cfg = args.cfg
    else:
        cfg = os.path.join(f'./datasets/{args.data}/models/vietcombank2.json')

    config.load_config(f'datasets/{args.data}/models/evisa.json')
    run(args.data,
        config.DatasetConfig.image_dir,
        config.DatasetConfig.label_file,
        f'./datasets/{args.data}/annotations/predict.csv',
        3000, 5000)
