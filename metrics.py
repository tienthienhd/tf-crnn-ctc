import tensorflow as tf
import numpy as np

import config


class CharacterAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='char_acc', **kwargs):
        super(CharacterAccuracy, self).__init__(name=name, **kwargs)
        self.char_acc = self.add_weight(name='char_acc', initializer='zeros')

    def update_state(self, y_true, y_pred):
        input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = tf.keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)[0][0][
                  :, :config.DatasetConfig.max_len
                  ]
        preds_idx = tf.cast(results, dtype=tf.int32)
        targets = tf.cast(y_true, dtype=tf.int32)
        const_rej_char = tf.constant(rej_char, shape=targets.get_shape())
        weights = tf.cast(tf.not_equal(targets, const_rej_char), dtype=tf.float32)
        correct_chars = tf.cast(tf.equal(preds_idx, targets), dtype=tf.float32)
        accuracy_per_example = tf.compat.v1.div(
            tf.reduce_sum(input_tensor=tf.multiply(
                correct_chars, weights), axis=1),
            tf.reduce_sum(input_tensor=weights, axis=1))
        char_acc_batch = tf.reduce_mean(input_tensor=accuracy_per_example)

        seq_acc_per_example = tf.cast(tf.equal(accuracy_per_example,
                                               tf.constant(1.0, shape=accuracy_per_example.get_shape())),
                                      dtype=tf.float32)
        seq_acc_batch = tf.reduce_mean(seq_acc_per_example)

        seq_acc.append(seq_acc_batch.numpy())

        char_acc.append(char_acc_batch.numpy())

        self.char_acc.

    def result(self):
        return self.char_acc

    def reset_states(self):
        self.char_acc = 0.0
