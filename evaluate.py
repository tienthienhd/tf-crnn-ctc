from tensorflow import keras
import os
import config
import numpy as np
from data_provider_tfrecord import get_data
import cv2

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
        chars = [idx2char[i] if (i != -1) else '' for i in indices]
    elif not check:
        chars = [idx2char[i - 2] if (i != -1) else '' for i in indices]
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
    result = decode_text(list(result.numpy()[0]), idx2char)
    label = decode_text(list(label), idx2char)
    return result, label


def predict_model(img, label):
    predict = load_model(path_best_model).predict(img)
    predict = decode_img(predict, label)
    return predict


def load_model_from_dataset(dataset):
    list_image, list_label, list_predict = [], [], []
    for data in dataset.as_numpy_iterator():
        image = (np.array(data['image'][0]) + 0.5) * 255.0
        image = image.astype(np.uint8)
        label = data['label'][0]
        predict_text, label_text = predict_model(image, label)
        list_image.append(image)
        list_label.append(label_text)
        list_predict.append(predict_text)
    return  list_image, list_label, list_label


if __name__ == "__main__":
    path_best_model = os.path.join(config.TrainingConfig.checkpoints, 'best_train_model.h5')
    dataset = get_data('test')
    img, lb, predict = load_model_from_dataset(dataset)
    acc = eval_model(lb, predict)
    print(acc)
    check_wrong_img(img, lb, predict)


