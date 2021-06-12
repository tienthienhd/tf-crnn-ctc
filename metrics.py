import numpy as np
import tensorflow as tf

import config

def char_acc(y_true, y_pred):
    input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = tf.keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)[0][0][
              :, :config.DatasetConfig.max_len
              ]
    preds_idx = tf.cast(results, dtype=tf.int32)
    targets = tf.cast(y_true, dtype=tf.int32)
    const_rej_char = tf.constant(-1, shape=targets.get_shape())
    weights = tf.cast(tf.not_equal(targets, const_rej_char), dtype=tf.float32)
    correct_chars = tf.cast(tf.equal(preds_idx, targets), dtype=tf.float32)
    accuracy_per_example = tf.compat.v1.div(
        tf.reduce_sum(input_tensor=tf.multiply(
            correct_chars, weights), axis=1),
        tf.reduce_sum(input_tensor=weights, axis=1))
    char_acc_batch = tf.reduce_mean(input_tensor=accuracy_per_example)
    return char_acc_batch

#
# print(char_acc(np.array([[3,6,1,1,7,2,9,8,7, -1 , -1, -1]]), np.load(open('test.npy', 'rb'))))

def seq_acc(y_true, y_pred):
    input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = tf.keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)[0][0][
              :, :config.DatasetConfig.max_len
              ]
    preds_idx = tf.cast(results, dtype=tf.int32)
    targets = tf.cast(y_true, dtype=tf.int32)
    const_rej_char = tf.constant(-1, shape=targets.get_shape())
    weights = tf.cast(tf.not_equal(targets, const_rej_char), dtype=tf.float32)
    correct_chars = tf.cast(tf.equal(preds_idx, targets), dtype=tf.float32)
    accuracy_per_example = tf.compat.v1.div(
        tf.reduce_sum(input_tensor=tf.multiply(correct_chars, weights), axis=1),
        tf.reduce_sum(input_tensor=weights, axis=1))
    char_acc_batch = tf.reduce_mean(input_tensor=accuracy_per_example)



class CharacterAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='char_acc', **kwargs):
        super(CharacterAccuracy, self).__init__(name=name, **kwargs)
        self.char_acc = self.add_weight(name='char_accs', initializer='zeros')
        self.update_metric = tf.Variable(False)

    def update_state(self, y_true, y_pred):
        if self.update_metric:
            input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
            # Use greedy search. For complex tasks, you can use beam search
            results = tf.keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)[0][0][
                      :, :config.DatasetConfig.max_len
                      ]
            preds_idx = tf.cast(results, dtype=tf.int32)
            targets = tf.cast(y_true, dtype=tf.int32)
            const_rej_char = tf.constant(-1, shape=targets.get_shape())
            weights = tf.cast(tf.not_equal(targets, const_rej_char), dtype=tf.float32)
            correct_chars = tf.cast(tf.equal(preds_idx, targets), dtype=tf.float32)
            accuracy_per_example = tf.compat.v1.div(
                tf.reduce_sum(input_tensor=tf.multiply(
                    correct_chars, weights), axis=1),
                tf.reduce_sum(input_tensor=weights, axis=1))
            char_acc_batch = tf.reduce_mean(input_tensor=accuracy_per_example)
            tf.print(char_acc_batch)

            self.char_acc.assign_add(char_acc_batch)

    def result(self):
        return self.char_acc

    def reset_states(self):
        self.char_acc.assign(0.0)


class ToggleMetrics(tf.keras.callbacks.Callback):
    '''On test begin (i.e. when evaluate() is called or
     validation data is run during fit()) toggle metric flag '''

    def on_test_begin(self, logs):
        for metric in self.model.metrics:
            if 'CharacterAccuracy' in metric.name:
                metric.update_metric.assign(True)

    def on_test_end(self, logs):
        for metric in self.model.metrics:
            if 'CharacterAccuracy' in metric.name:
                metric.update_metric.assign(False)
