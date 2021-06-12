import os

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend


def block0(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """
    A basic residual block.
    :param x: input tensor
    :param filters: integer, filters of the layer
    :param kernel_size: default 3, kernel size of the layer
    :param stride: default 1, stride of the first layer;
    :param conv_shortcut: default False, use convolution shortcut if True, otherwise identity shortcut.
    :param name: string, block label.
    :return: Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(filters, 1, stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, kernel_size, stride, padding='same', name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, 1, padding='same', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])

    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack0(x, filters, blocks, stride1=(2, 1), name=None):
    """
    A set of stacked basic residual blocks.
    :param x: input tensor
    :param filters: integer, filters of layer in a block
    :param blocks: integer, blocks in the stacked blocks
    :param stride1: default 2, stride of the first layer in the first block.
    :param name: string, stack label
    :return: Output tensor for the stacked blocks
    """
    x = block0(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block0(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def ResNet(stack_fn, preact, model_name='resnet', input_tensor=None,
           input_shape=None, **kwargs):
    """
    instantiates the ResNet, ResNetV2, and ResNeXt architecture
    Optionally loads weights pre-trained on ImageNet.
    :param stack_fn: a function that returns output tensor for the stacked residual blocks
    :param preact: whether to use pre-activation or not (True for ResNetV2, False for ResNet and ResNeXt)
    :param model_name: string, model name
    :param input_tensor: optional Keras tensor (i.e. output of 'layers.Input()') to be use as image input for the model.
    :param input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
    :param kwargs:
    :return: a keras model instance
    :raises
        ValueError: in case of invalid argument for 'weights', or invalid input shape
    """

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        img_input = input_tensor
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, name='conv1_conv')(x)


    if preact is False:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    # Create Model
    model = models.Model(img_input, x, name=model_name)

    return model


def ResNet18(input_tensor=None, input_shape=None, **kwargs):
    def stack_fn(x):
        x = stack0(x, 64, 2, stride1=1, name='conv2')
        x = stack0(x, 128, 2, name='conv3')
        x = stack0(x, 256, 2, name='conv4')
        x = stack0(x, 512, 2, name='conv5')
        return x

    return ResNet(stack_fn, preact=False, model_name='resnet18', input_tensor=input_tensor, input_shape=input_shape,
                  **kwargs)
