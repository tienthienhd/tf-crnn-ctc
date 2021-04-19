import os
dataset = {
    'data_name': 'plate_2',
    'image_dir': '/media/data_it/Data_set/database_image/card/vr/info/train/plate_new',
    'label_file': '/media/data_it/Data_set/database_image/card/vr/info/train/vr_plate_label_train.csv',

    'data_dir': '/media/thiennt/projects/cv_end_to_end/training/ocr/crnn_ctc/datasets/vr_plate',
    'train_file_patterns': ['*_train.records*'],
    'test_file_patterns': ['*_test.records*'],
    'height': 32,
    'max_width': 512,
    'charset': sorted('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-'),
    'max_len': 11,
    'test_size': 0.1
}


cnn = [
    {'filter': 32, 'kernel_size': 3, 'padding': 'same', 'strides': 1, 'pool_size': 2, 'padding_pool': 'valid', 'batch_norm': True},
    {'filter': 64, 'kernel_size': 3, 'padding': 'same', 'strides': 1, 'pool_size': 2, 'padding_pool': 'valid', 'batch_norm': True},
    {'filter': 128, 'kernel_size': 3, 'padding': 'same', 'strides': 1, 'pool_size': 2, 'padding_pool': 'valid', 'batch_norm': True},
]

rnn = [
    {'units': 128},
    {'units': 64}
]

head = {
    'classes': len(dataset['charset']) + 1
}

training = {
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'epochs': 5,
    'checkpoint': os.path.join(dataset['data_dir'], 'checkpoint'),
    'batch_size': 32,
}

