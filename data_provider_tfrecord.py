import os

import tensorflow as tf

import config


def get_data(data: str = 'train') -> tf.data.Dataset:
    """
    Args:
      data: train/test

    Returns:
      dataset : tf.data.Dataset object.
                elements structured as [features, labels]
                Example feature structure can be seen in postbatch_fn
    """

    # Get raw data
    if data == 'train':
        file_patterns = config.DatasetConfig.train_file_patterns
    else:
        file_patterns = config.DatasetConfig.test_file_patterns
    dataset = get_dataset(config.DatasetConfig.data_dir, file_patterns)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Preprocess data
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Prepare dataset for Estimator ingestion
    # ie: sparsify labels for CTC operations (eg loss, decoder)
    # and convert elements to be [features, label]
    dataset = dataset.map(postbatch_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # dataset = dataset.repeat(config.training['epochs'])

    dataset = dataset.padded_batch(config.TrainingConfig.batch_size,
                                   padded_shapes={
                                       'image': [config.DatasetConfig.height, None, config.DatasetConfig.depth],
                                       'label': [None]
                                   },
                                   padding_values={
                                       'image': 0.0,
                                       'label': len(config.DatasetConfig.charset)
                                   })
    # dataset = dataset.batch(config.TrainingConfig.batch_size)

    return dataset


def get_dataset(base_dir, file_patterns):
    """ Get a Dataset from TFRecord files.
    Parameters:
      base_dir      : Directory containing the TFRecord files
      file_patterns : List of wildcard patterns for TFRecord files to read
    Returns:
      image   : preprocessed image
                  tf.float32 tensor of shape [32, ?, 1] (? = width)
      width   : width (in pixels) of image
                  tf.int32 tensor of shape []
      labels  : list of indices of characters mapping text->out_charset
                  tf.int32 tensor of shape [?] (? = length+1)
      length  : length of labels (sans -1 EOS token)
                  tf.int32 tensor of shape []
      text    : ground truth string
                  tf.string tensor of shape []
    """
    # Get filenames as list of tensors
    tensor_filenames = _get_filenames(base_dir, file_patterns)

    # Get filenames into a dataset format
    ds_filenames = tf.data.Dataset.from_tensor_slices(tensor_filenames)

    dataset = tf.data.TFRecordDataset(ds_filenames,
                                      num_parallel_reads=tf.data.experimental.AUTOTUNE)

    return dataset


# Create a generator
rng = tf.random.Generator.from_seed(123, alg='philox')


def preprocess_fn(data):
    """Parse the elements of the dataset"""

    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/labels': tf.io.VarLenFeature(dtype=tf.int64),
        'image/width': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=1),
        'image/filename': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'text/string': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'text/length': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=1)
    }

    features = tf.io.parse_single_example(data, feature_map)
    # Initialize fields according to feature map

    # Convert to grayscale
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3 if config.DatasetConfig.depth == 3 else 1)

    width = tf.cast(features['image/width'], tf.int32)  # for ctc_loss
    # label = tf.io.serialize_sparse(features['image/labels'])  # for batching
    label = tf.cast(features['image/labels'], tf.int32)
    length = features['text/length']
    text = features['text/string']

    image = preprocess_image(image)

    seed = rng.make_seeds(2)[0]
    if config.TrainingConfig.augment_data:
        image = augment_image(image, seed)

    return image, width, label, length, text


def postbatch_fn(image, width, label, length, text):
    """Post-batching, postprocessing: packs raw tensors into a dictionary for
       Dataset's iterator output"""

    # Batching is complete, so now we can re-sparsify our labels for ctc_loss
    label = tf.sparse.to_dense(label)

    # Format relevant features for estimator ingestion
    features = {
        "image": image,
        "width": width,
        "length": length,
        "text": text
    }

    # return image, label
    return {'image': image, 'label': label}


def _get_filenames(base_dir, file_patterns=['*.tfrecord']):
    """Get a list of record files"""

    # List of lists ...
    data_files = [tf.io.gfile.glob(os.path.join(base_dir, file_pattern))
                  for file_pattern in file_patterns]
    # flatten
    data_files = [data_file for sublist in data_files for data_file in sublist]
    print(f'Load data from: {data_files}')

    return data_files


def preprocess_image(image):
    """Preprocess image: Rescale and fix image height"""

    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype(image, tf.float32)
#     image = tf.subtract(image, 0.5)

    image = tf.image.resize(image, [config.DatasetConfig.height, config.DatasetConfig.max_width],
                            preserve_aspect_ratio=True)

    # Pad with copy of first row to expand to 32 pixels height
    # first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
    # image = tf.concat([first_row, image], 0)

    return image


def augment_image(image, seed):
    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)
    image = tf.image.stateless_random_contrast(image, lower=0.4, upper=0.6, seed=new_seed)
    return image


if __name__ == '__main__':
    data_name = 'csgt'
    config.load_config(f'./datasets/{data_name}/models/config.json')
    dataset = get_data('test')
    print(dataset)

    char_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=list(config.DatasetConfig.charset), num_oov_indices=0, mask_token=None
    )

    num_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    import matplotlib.pyplot as plt
    import numpy as np

    for data in dataset.as_numpy_iterator():
        image = (np.array(data['image'][0]) + 0.5) * 255.0
        image = image.astype(np.uint8)
        plt.imshow(image)
        plt.title(tf.strings.join(num_to_char(data['label'][0])).numpy().decode('utf-8'))
        plt.show()
