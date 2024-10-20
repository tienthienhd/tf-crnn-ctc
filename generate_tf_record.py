import argparse
import os
import string
import sys

import contextlib
import cv2
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import config

"""Each record within the TFRecord file is a serialized Example proto. 
The Example proto contains the following fields:
  image/encoded: string containing JPEG encoded grayscale image
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/filename: string containing the basename of the image file
  image/labels: list containing the sequence labels for the image text
  text/string: string specifying the human-readable version of the text
  text/length: integer, length of text
"""


def create_tf_record(image_dir: str, df_label: pd.DataFrame, characters: str, output_path: str, num_shards: int = 1,
                     des: str = "Process"):
    """
    Read image, label and
    Generate tf record file
    Args:
        characters:
        df_label: Dataframe, dataframe label: filename, label
        image_dir: string, path of image
        output_path: string, path to output file
        num_shards: number of output file shards
        des: description of progressing

    Returns:

    """
    print('===========================================================================')
    print(f'Image dir: {image_dir}')
    print(f'Number of Example: {df_label.shape[0]}')
    print(f'Output {num_shards} file: {output_path}')
    print(f'Charset: {characters}')

    char_statistic = {}
    for c in characters:
        char_statistic[c] = 0
    all_labels = df_label['label'].tolist()
    for label in all_labels:
        for c in label:
            if c not in char_statistic:
                print(f"Error: {label}", file=sys.stderr)
            char_statistic[c] += 1
    char_statistic = {k: v for k, v in sorted(char_statistic.items(), key=lambda item: item[1])}

    print(f'Charset statistic: {char_statistic}')
    print(f'Charset actual: {"".join(sorted(char_statistic.keys()))}')
    print('===========================================================================')

    with contextlib.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(tf_record_close_stack, output_path, num_shards)

        # # Mapping characters to integers
        # char_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(
        #     vocabulary=list(characters), num_oov_indices=0, mask_token=None
        # )
        #
        # # Mapping integers back to original characters
        # num_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(
        #     vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
        # )
        #
        # print(f'Vocab Mapping: {char_to_num.get_vocabulary()}')
        # print(f'Vocab size: {char_to_num.vocab_size()}')
        # print(f'Oov token: {char_to_num.oov_token}')

        for idx, row in tqdm(df_label.iterrows(), total=len(df_label), ncols=100, desc=des):
            filename = row['filename']
            filename = os.path.basename(filename)
            label = row['label']

            # print(f"Process {idx}: {filename}\t-\t{label}", end='\r')
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(row)
            h, w, c = image.shape
            image_data = open(image_path, 'rb').read()
            # labels = char_to_num(tf.strings.unicode_split(label, input_encoding='UTF-8')).numpy().tolist()
            try:
                labels = [characters.index(c) for c in label]
            except Exception as e:
                print(row)
                raise e
            example = create_tf_example(filename, image_data, labels, label, h, w)
            shard_idx = idx % num_shards
            if example:
                output_tfrecords[shard_idx].write(example.SerializeToString())


def create_tf_example(filename, image_data, labels, text, height, width) -> tf.train.Example:
    """
    Build an example proto for an example
    Args:
        filename: string, path to an image file
        image_data: bytes, JPEG encoding of image
        labels: integer list, identifiers for ground truth for the network
        text: string, unique human-readable
        height: integer, image height in pixels
        width: integer, image width in pixels

    Returns:
        Examples proto
    """
    feature_dict = {
        'image/filename': bytes_feature(tf.compat.as_bytes(filename)),
        'image/encoded': bytes_feature(image_data),
        'image/labels': int64_list_feature(labels),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'text/string': bytes_feature(tf.compat.as_bytes(text)),
        'text/length': int64_feature(len(text))

    }
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict)
    )
    return example


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.

    Args:
      exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
      base_path: The base path for all shards
      num_shards: The number of shards

    Returns:
      The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def filter_not_in_charset(df_label, charset=string.ascii_letters + string.digits):
    records = []
    for idx, row in df_label.iterrows():
        l = row['label']
        if all([c in charset for c in l]):
            records.append({
                "filename": row["filename"],
                "label": row["label"]
            })
    return pd.DataFrame.from_records(records)
    # return df_label[~df_label['Column1'].str.contains('a')]


def main():
    df_label = pd.read_csv(config.DatasetConfig.label_file, header=0, dtype={'filename': str, 'label': str},
                           keep_default_na=False, na_values=[''])
    if num > 0:
        df_label = df_label.iloc[0: num]

    if filter_charset:
        df_label = filter_not_in_charset(df_label, config.DatasetConfig.charset)
    test_size = config.DatasetConfig.test_size
    if test_size < 1:
        test_size = test_size * len(df_label)
    test_size = int(test_size)
    df_label_test = df_label.iloc[-test_size:, :]
    df_label_train = df_label.iloc[:-test_size, :]

    output_path = os.path.join(config.DatasetConfig.data_dir, f'{config.DatasetConfig.data_name}_train.records')
    create_tf_record(config.DatasetConfig.image_dir, df_label_train, config.DatasetConfig.charset, output_path,
                     des='Train dataset')

    output_path = os.path.join(config.DatasetConfig.data_dir, f'{config.DatasetConfig.data_name}_test.records')
    create_tf_record(config.DatasetConfig.image_dir, df_label_test, config.DatasetConfig.charset, output_path,
                     des='Test dataset')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="evisa")
    parser.add_argument('--num', type=int, default="0")
    parser.add_argument('--filter_charset', type=bool, default="true")
    parser.add_argument('--cfg', type=str, default="evisa.json")

    args = parser.parse_args()

    if args.cfg != 'evisa.json':
        cfg = args.cfg
    else:
        cfg = os.path.join(f'./datasets/{args.data}/models/evisa.json')
    num = args.num
    filter_charset = args.filter_charset
    config.load_config(cfg)
    main()
