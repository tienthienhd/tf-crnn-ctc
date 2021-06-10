import csv
from typing import List, Dict

import pandas as pd
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

normalized_levenshtein = NormalizedLevenshtein()


def evaluate(actual: List[str], predict: List[str], is_lower=False) -> Dict:
    count_seq_true = 0
    ed_score = 0.0
    wrong_label = []
    wrong_pred = []
    for label, pred in zip(actual, predict):
        if is_lower:
            label = label.lower()
            pred = pred.lower()
        sim = normalized_levenshtein.similarity(label, pred)
        if sim == 1:
            count_seq_true += 1
        else:
            wrong_label.append(label)
            wrong_pred.append(pred)
        ed_score += sim

    seq_acc = count_seq_true / len(actual)
    char_acc = ed_score / len(actual)
    print('======================= Evaluate ==========================')
    print(f'Number of sample: {len(actual)}')
    print(f'Sequence Accuracy: {seq_acc}')
    print(f'Character Accuracy: {char_acc}')
    print('Wrong predict: ')
    for l, p in zip(wrong_label[:10], wrong_pred[:10]):
        print('------------------------')
        print(f'{l:30}')
        print(f'{p:30}')
    print('===========================================================')
    return {
        'seq_acc': seq_acc,
        'char_acc': char_acc
    }


def eval_from_csv(label_path: str, predict_path: str):
    df_label = pd.read_csv(label_path, header=0, dtype={'filename': str, 'label': str})
    df_pred = pd.read_csv(predict_path, header=0, dtype={'filename': str, 'pred': str},
                          names=['filename', 'pred', 'conf'])
    df = pd.merge(left=df_label, right=df_pred, left_on='filename', right_on='filename')
    labels = df['label'].tolist()
    preds = df['pred'].tolist()
    evaluate(labels, preds, is_lower=True)


def drop_not_exists():
    image_dir = '/media/data_it/Data_set/database_image/card/id/words/BIRTHDAY'
    df = pd.read_csv('/media/data_it/Data_set/database_image/card/id/words/birthday.csv', header=0,
                     dtype={'filename': object, 'label': object})
    print(df.shape)
    import os
    count_drop = 0
    for idx, row in df.iterrows():
        if not os.path.exists(os.path.join(image_dir, row['filename'])):
            print('drop', row['filename'])
            count_drop += 1
            df.drop(idx, axis=0, inplace=True)
    print(df.shape)
    print('drop', count_drop)
    df.to_csv('/media/data_it/Data_set/database_image/card/id/words/birthday_2.csv', index=False, quoting=csv.QUOTE_ALL)


if __name__ == '__main__':
    label_path = '/media/data_it/Data_set/database_image/card/id/words/address_eval_5k_from_0.csv'
    predict_path = '/media/data_it/Data_set/database_image/card/id/words/address_v2_pred_070621.csv'
    eval_from_csv(label_path, predict_path)
