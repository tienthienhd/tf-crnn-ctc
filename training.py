import argparse
import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from model import build_model
import config
from data_provider_tfrecord import get_data


class ModelCheckpointCallback(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, meta_file, **kwargs):
        super().__init__(filepath, **kwargs)
        self.metadata_file = meta_file

    def on_epoch_end(self, epoch, logs=None):
        super(ModelCheckpointCallback, self).on_epoch_end(epoch, logs)
        with open(self.metadata_file, 'w') as f:
            f.write(json.dumps({"epochs": epoch}))


def run():
    # Prepare dataset for training
    train_dataset = get_data(data="train")
    val_dataset = get_data(data="test")

    # Build model
    model, training_model = build_model()
    training_model.summary()
    # tf.keras.utils.plot_model(training_model, show_shapes=True, show_dtype=True, expand_nested=True,
    #                           show_layer_names=True)
    initial_epoch = 0
    best_model_path = os.path.join(config.TrainingConfig.checkpoints, 'best_train_model.h5')
    last_model_inference = os.path.join(config.TrainingConfig.checkpoints, 'last_inference_model.h5')
    meta_file = os.path.join(config.TrainingConfig.checkpoints, 'checkpoint')
    if os.path.exists(best_model_path):
        training_model.load_weights(best_model_path)
        if os.path.exists(meta_file):
            meta = json.load(open(meta_file, 'r'))
            initial_epoch = meta['epochs']

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.TensorBoard(config.TrainingConfig.checkpoints),
        ModelCheckpointCallback(best_model_path, meta_file, save_best_only=True)
    ]

    history = training_model.fit(train_dataset,
                                 validation_data=val_dataset,
                                 validation_freq=1,
                                 callbacks=callbacks,
                                 batch_size=config.TrainingConfig.batch_size,
                                 initial_epoch=initial_epoch,
                                 epochs=config.TrainingConfig.epochs)

    # Save model
    model.save(last_model_inference, include_optimizer=False)

    # plot history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='vr_plate')
    parser.add_argument('--cfg', type=str, default="config.json")

    args = parser.parse_args()

    if args.cfg != 'config.json':
        cfg = args.cfg
    else:
        cfg = os.path.join(f'./datasets/{args.data}/config.json')

    config.load_config(cfg)
    run()
