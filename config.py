import os


class DatasetConfig:
    data_name = 'plate_2'
    image_dir = '/media/data_it/Data_set/database_image/card/vr/info/train/plate_new'
    label_file = '/media/data_it/Data_set/database_image/card/vr/info/train/vr_plate_label_train.csv'

    data_dir = '/media/thiennt/projects/cv_end_to_end/training/ocr/crnn_ctc/datasets/vr_plate'
    train_file_patterns = ['*_train.records*']
    test_file_patterns = ['*_test.records*']
    height = 32
    max_width = 512
    charset = sorted('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-')
    max_len = 11
    test_size = 0.


class CnnConfig:
    type_model = 'stack'
    layers = [
        {'filter': 64, 'kernel_size': 3, 'padding': 'same', 'strides': 1, 'pool_size': 2, 'padding_pool': 'valid',
         'batch_norm': True},
        {'filter': 128, 'kernel_size': 3, 'padding': 'same', 'strides': 1, 'pool_size': 2, 'padding_pool': 'valid',
         'batch_norm': True},
    ]


class RnnConfig:
    input_depth = 64
    input_dropout = 0.2
    rnn_layers = [
        {'units': 128, 'dropout': 0.25},
        {'units': 64, 'dropout': 0.25}
    ]


class HeadConfig:
    classes = len(DatasetConfig.charset) + 1


class TrainingConfig:
    optimizer = 'adam'
    learning_rate = 0.001
    epochs = 10
    checkpoints = os.path.join(DatasetConfig.data_dir, 'checkpoint')
    batch_size = 64
