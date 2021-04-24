import os
import matplotlib.pyplot as plt
import tensorflow as tf
from model import build_model
import config
from data_provider_tfrecord import get_data


def run():
    # Prepare dataset for training
    train_dataset = get_data(data="train")
    val_dataset = get_data(data="test")

    # Build model
    model, training_model = build_model()
    training_model.summary()
    # tf.keras.utils.plot_model(training_model, show_shapes=True, show_dtype=True, expand_nested=True,
    #                           show_layer_names=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.TensorBoard(config.TrainingConfig.checkpoints, write_images=True, update_freq='batch'),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(config.TrainingConfig.checkpoints, "best_model.h5"),
                                           save_best_only=True, save_freq=100)
    ]

    history = training_model.fit(train_dataset,
                                 validation_data=val_dataset,
                                 validation_steps=100,
                                 validation_freq=1,
                                 callbacks=callbacks,
                                 batch_size=config.TrainingConfig.batch_size,
                                 epochs=config.TrainingConfig.epochs)

    # Save model
    model.save(os.path.join(config.TrainingConfig.checkpoints, "inference_model.h5"), include_optimizer=False)

    # plot history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    run()
