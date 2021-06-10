import tensorflow as tf
import config
from backbone.resnet import ResNet18
from data_provider_tfrecord import get_data


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64", name='ctc_cast_batch_len')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64", name='ctc_cast_input_len')
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64", name='ctc_cast_label_len')

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def get_cnn_feature(x):
    if config.CnnConfig.type_model == 'stack':
        for i, block in enumerate(config.CnnConfig.layers):
            x = tf.keras.layers.Conv2D(filters=block['filter'], kernel_size=block['kernel_size'], padding=block['padding'],
                                       strides=block['strides'], name=f'conv_{i}')(x)
            x = tf.keras.layers.MaxPooling2D(block['pool_size'], padding=block['padding_pool'], name=f'max_pool_{i}')(x)
            if block['batch_norm']:
                x = tf.keras.layers.BatchNormalization(name=f'bn_{i}')(x)
        return x
    elif config.CnnConfig.type_model == 'resnet_18':
        model = ResNet18(x)
        return model.output


def get_rnn_feature(x):
    """
    Feed cnn feature to B LSTM
    Args:
        x: tensor, shape=(b, h, w, c)

    Returns:

    """
    x = tf.keras.layers.Permute(dims=(2, 1, 3))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.Dense(config.RnnConfig.input_depth, activation="relu")(x)
    x = tf.keras.layers.Dropout(config.RnnConfig.input_dropout)(x)

    for i, block in enumerate(config.RnnConfig.layers):
        if 'dropout' not in block:
            block['dropout'] = 0.0
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=block['units'],
                                 return_sequences=True,
                                 dropout=block['dropout']),
            name=f'BLSTM_{i}')(x)
    return x


def get_output(x):
    outputs = tf.keras.layers.Dense(units=config.HeadConfig.classes, activation='softmax', name='outputs')(x)
    return outputs


def build_model():
    img_input = tf.keras.layers.Input(shape=(config.DatasetConfig.height, None, config.DatasetConfig.depth),
                                      dtype=tf.float32, name='image')
    label = tf.keras.layers.Input(shape=(None,), dtype=tf.float32, name='label')

    x = get_cnn_feature(img_input)
    x = get_rnn_feature(x)
    predict = get_output(x)

    model = tf.keras.Model(inputs=img_input, outputs=predict, name='inference_model')

    ctc_output = CTCLayer(name='ctc_loss')(label, predict)

    train_model = tf.keras.Model(inputs=[img_input, label], outputs=ctc_output, name='train_model')

    train_model.compile(config.TrainingConfig.optimizer)

    return model, train_model


if __name__ == '__main__':
    model, train_model = build_model()
    model.summary()
    train_model.summary()

    i1 = tf.random.uniform(shape=[2, 32, 128, 3])
    i2 = tf.random.uniform(shape=[2, 32, 64, 3])
    val_dataset = get_data('test')  # .batch(config.training['batch_size'])
    print(val_dataset)


    def test(data):
        return data['image']


    a = val_dataset.map(test).padded_batch(4, [32, None, 3], 0.0).take(2).as_numpy_iterator()
    a = list(a)

    # import cv2
    # print(a)
    # for image in a:
    #     image = image.astype(np.uint8)
    #     for i in image:
    #         print(i.shape)
    #         cv2.imshow("test", i)
    #         cv2.waitKey(0)

    a = model.predict_on_batch(a)
    print(a)
