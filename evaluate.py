import argparse
import os
import config
import numpy as np
import cv2
from tensorflow import keras
from data_provider_tfrecord import get_data
from itertools import chain
from asrtoolkit import cer, wer


def load_model(path):
    model = keras.models.load_model(path)
    return model


def eval_model_words(label: list, predict: list) -> float:
    count = 0
    total = len(label)
    for idx in range(total):
        if label[idx] == predict[idx]:
            count += 1
    acc = round(count / total, 2) * 100
    return acc

def eval_model_character(orig_texts, predict):
    cer_ = 0
    for idx in range(len(predict)):
        cer_i = cer(orig_texts[idx], predict[idx])
        t = cer_i / 100
        cer_ += t
    return round((1-cer_/len(orig_texts))*100, 2)


def check_wrong_img(img: list, label: list, predict: list):
    for i in range(len(label)):
        if label[i] != predict[i]:
            cv2.imshow(f"Wrong_{i}: {predict[i]} Correct_{i}: {label}", img[i])
            cv2.waitKey()
            print(f"Wrong_{i}: {predict[i]} Correct_{i}: {label}")


def decode_text(indices, idx2char, check=True):
    chars = None
    if check:
        chars = [idx2char[i] if i in idx2char.keys() else '' for i in indices]
    elif not check:
        chars = [idx2char[i - 2] if i in idx2char.keys() else '' for i in indices]
    return "".join(chars)


def decode_img(predict, label):
    vocab = config.DatasetConfig.charset
    max_len = config.DatasetConfig.max_len
    char2idx = {}
    idx2char = {}
    for i, c in enumerate(vocab):
        char2idx[c] = i
        idx2char[i] = c
    input_len = np.ones(predict.shape[0]) * predict.shape[1]
    result = keras.backend.ctc_decode(predict, input_length=input_len, greedy=True)[0][0][:, :max_len]
    label = [decode_text(lb.tolist(), idx2char) for lb in label]
    result = [decode_text(list(rlt.numpy()), idx2char) for rlt in result]
    return result, label


def predict_model(model, img, label):
    # img = preprocessing_data(img)
    predict = model.predict(img)
    predict, lb = decode_img(predict, label)
    return predict, lb


def load_model_from_dataset(dataset):
    model = load_model(path_best_model)
    list_image, list_label, list_predict = [], [], []
    i = 0
    for data in dataset.as_numpy_iterator():
        i += 1
        image = data['image']
        label = data['label']
        predict_text, label_text = predict_model(model, image, label)
        list_image.append(image)
        list_label.append(label_text)
        list_predict.append(predict_text)
    return list_image, list_label, list_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="vietcombank")
    parser.add_argument("--cfg", type=str, default="config.json")
    args = parser.parse_args()
    if args.cfg != 'config.json':
        cfg = args.cfg
    else:
        cfg = os.path.join(f'./datasets/{args.data}/models/config.json')
    config.load_config(cfg)
    path_best_model = os.path.join(config.TrainingConfig.checkpoints, f'{config.DatasetConfig.data_name}.h5')
    dataset = get_data('test')
    img, lb, predict = load_model_from_dataset(dataset)
    lb = list(chain.from_iterable(lb))
    predict = list(chain.from_iterable(predict))
    acc_w = eval_model_words(lb, predict)
    acc_c = eval_model_character(lb, predict)
    print(f"label: {lb}\n predict: {predict}\n {len(lb)}")
    print(f"Accuracy words: {acc_w}%")
    print(f"Accuracy character: {acc_c}%")
    # check_wrong_img(img, lb, predict)

