import json
import os


class DatasetConfig:
    data_name = 'plate_2'
    image_dir = '/media/data_it/Data_set/database_image/card/vr/info/train/plate_new'
    label_file = '/media/data_it/Data_set/database_image/card/vr/info/train/vr_plate_label_train.csv'
    work_dir = '/media/thiennt/projects/cv_end_to_end/training/ocr/crnn_ctc/datasets/vr_plate'
    data_dir = os.path.join(work_dir, 'annotations')
    train_file_patterns = ['*_train.records*']
    test_file_patterns = ['*_test.records*']
    height = 32
    depth = 1
    max_width = 512
    charset = sorted('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-')
    max_len = 11
    test_size = 0.0
    normalize = True


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
    layers = [
        {'units': 128, 'dropout': 0.25},
        {'units': 64, 'dropout': 0.25}
    ]


class HeadConfig:
    classes = len(DatasetConfig.charset) + 1


class TrainingConfig:
    augment_data = False
    optimizer = 'adam'
    learning_rate = 0.001
    epochs = 10
    checkpoints = os.path.join(DatasetConfig.work_dir, 'models')
    batch_size = 64


def load_config(config_path: str):
    config = json.load(open(config_path, 'r'))

    data_config = config['DatasetConfig']
    DatasetConfig.data_name = data_config['data_name']
    DatasetConfig.image_dir = data_config['image_dir']
    DatasetConfig.label_file = data_config['label_file']
    DatasetConfig.work_dir = data_config['work_dir']
    DatasetConfig.data_dir = os.path.join(DatasetConfig.work_dir, 'annotations')
    DatasetConfig.train_file_patterns = data_config['train_file_patterns']
    DatasetConfig.test_file_patterns = data_config['test_file_patterns']
    DatasetConfig.height = data_config['height']
    DatasetConfig.depth = data_config['depth']
    DatasetConfig.max_width = data_config['max_width']
    DatasetConfig.charset = sorted(data_config['charset'])
    DatasetConfig.max_len = data_config['max_len']
    DatasetConfig.test_size = data_config['test_size']
    DatasetConfig.normalize = data_config.get('normalize', True)

    cnn_config = config['CnnConfig']
    CnnConfig.type_model = cnn_config['type_model']
    CnnConfig.layers = cnn_config.get('layers')

    rnn_config = config['RnnConfig']
    RnnConfig.input_depth = rnn_config['input_depth']
    RnnConfig.input_dropout = rnn_config['input_dropout']
    RnnConfig.layers = rnn_config['layers']

    HeadConfig.classes = len(DatasetConfig.charset) + 1

    training_config = config['TrainingConfig']
    TrainingConfig.augment_data = training_config['augment_data']
    TrainingConfig.optimizer = training_config['optimizer']
    TrainingConfig.learning_rate = training_config['learning_rate']
    TrainingConfig.epochs = training_config['epochs']
    TrainingConfig.checkpoints = os.path.join(DatasetConfig.work_dir, 'models')
    TrainingConfig.batch_size = training_config['batch_size']
