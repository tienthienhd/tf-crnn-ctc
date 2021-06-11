from tensorflow import keras
import os
import config
import numpy as np
from data_provider_tfrecord import get_data
import cv2
from numpy import expand_dims
import tensorflow as tf


def load_model(path):
    model = keras.models.load_model(path)
    return model


def eval_model(label: list, predict: list) -> float:
    count = 0
    total = len(label)
    for idx in range(total):
        if label[idx] == predict[idx]:
            count += 1
    acc = round(count / total, 2) * 100
    return acc


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


def decode_img(predict: list, label: list):
    vocab = config.DatasetConfig.charset
    max_len = config.DatasetConfig.max_len
    char2idx = {}
    idx2char = {}
    for i, c in enumerate(vocab):
        char2idx[c] = i
        idx2char[i] = c
    input_len = np.ones(predict.shape[0]) * predict.shape[1]
    result = keras.backend.ctc_decode(predict, input_length=input_len, greedy=True)[0][0][:, :max_len]
    result = decode_text(list(result.numpy()[0]), idx2char)
    label = decode_text(label, idx2char)
    return result, label


def predict_model(model, img, label):
    img = preprocessing_data(img)
    predict = model.predict(img)
    predict, lb = decode_img(predict, label)
    return predict, lb


def add_padding(img, img_h, img_w):
    img = img
    hh, ww, cc = img.shape
    try:
        rate = ww / hh
        img = cv2.resize(img, (round(rate * img_h), img_h),
                         interpolation=cv2.INTER_AREA)
        color = (0, 0, 0)
        result = np.full((img_h, img_w, cc), color, dtype=np.uint8)
        result[:img_h, :round(rate * img_h)] = img
    except:
        result = cv2.resize(img, (img_w, img_h))
    return result


def preprocessing_data(img):
    h, w, c = img.shape
    img = add_padding(img, h, w)
    img = img / 255.0
    img = img.reshape((1, h, w))
    img = expand_dims(img, axis=-1)
    return img


def load_model_from_dataset(dataset):
    model = load_model(path_best_model)
    list_image, list_label, list_predict = [], [], []
    for data in dataset.as_numpy_iterator():
        image = (np.array(data['image'][0]) + 0.5) * 255.0
        image = image.astype(np.uint8)
        label = list(data['label'][0])
        # image = preprocessing_data(image, height, width)
        predict_text, label_text = predict_model(model, image, label)
        list_image.append(image)
        list_label.append(label_text)
        list_predict.append(predict_text)
    return list_image, list_label, list_predict


if __name__ == "__main__":
    path_best_model = os.path.join(config.TrainingConfig.checkpoints, 'last_inference_model.h5')
    dataset = get_data('test')
    img, lb, predict = load_model_from_dataset(dataset)
    acc = eval_model(lb, predict)
    print(f"label: {lb}\n predict: {predict}")
    print(f"Accuracy words: {acc}%")
    # check_wrong_img(img, lb, predict)
