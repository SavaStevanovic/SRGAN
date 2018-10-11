import tensorflow as tf
import numpy as np


def prelu(x, alpha):
    return tf.nn.relu(x)+alpha*(x-tf.abs(x))*0.5


def upscale_layer(input_tensor, name,
                  kernel_size, n_output_channels,
                  padding_mode='SAME', strides=(1, 1, 1, 1), alpha=0.2):

    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]

        weights_shape = list(kernel_size)+[n_input_channels, n_output_channels]
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        print(weights)

        upscale_layer = tf.nn.conv2d(input=input_tensor, filter=weights,
                                     strides=strides, padding=padding_mode)
        print(upscale_layer)

        upscale_layer = tf.depth_to_space(
            upscale_layer, 2, name='upscale_layer_preactivation')
        print(upscale_layer)

        upscale_layer = tf.identity(
            prelu(upscale_layer, alpha), name='activation')
        print(upscale_layer)

        return upscale_layer


def residual_block(input_tensor, name,
                   kernel_size, n_output_channels,
                   padding_mode='SAME', strides=(1, 1, 1, 1), alpha=0.2):

    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]

        weights_shape = list(kernel_size)+[n_input_channels, n_output_channels]
        weights1 = tf.get_variable(name='_weights1', shape=weights_shape)
        weights2 = tf.get_variable(name='_weights2', shape=weights_shape)

        print(weights1)

        # biases = tf.get_variable(
        #     name='_biases', initializer=tf.zeros(shape=[n_output_channels]))
        # print(biases)

        residual_block = tf.nn.conv2d(input=input_tensor, filter=weights1,
                                      strides=strides, padding=padding_mode)
        print(residual_block)

        # residual_block = tf.nn.bias_add(value=residual_block, bias=biases,
        #                                 name='net_pre-activation')
        # print(residual_block)

        residual_block = batch_normalize(residual_block)

        residual_block = tf.identity(
            prelu(residual_block, alpha), name='activation')
        print(residual_block)

        residual_block = tf.nn.conv2d(input=residual_block, filter=weights2,
                                      strides=strides, padding=padding_mode)
        print(residual_block)

        # residual_block = tf.nn.bias_add(value=residual_block, bias=biases,
        #                                 name='net_post-activation')
        # print(residual_block)
        residual_block = batch_normalize(residual_block)

        residual_block = tf.add(
            residual_block, input_tensor, name='output')
        print(residual_block)

        return residual_block


def batch_normalize(residual_block):
    mean, var = tf.nn.moments(residual_block, axes=[0, 1, 2])
    residual_block = tf.nn.batch_normalization(
        residual_block, mean, var, offset=0, name='normalized_post-activation')
    print(residual_block)
    return residual_block


def conv_layer(input_tensor, name,
               kernel_size, n_output_channels,
               padding_mode='SAME', strides=(1, 1, 1, 1)):

    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]

        weights_shape = list(kernel_size)+[n_input_channels, n_output_channels]
        weights = tf.get_variable(name='_weights', shape=weights_shape)

        print(weights)

        # biases = tf.get_variable(
        #     name='_biases', initializer=tf.zeros(shape=[n_output_channels]))
        # print(biases)

        conv = tf.nn.conv2d(input=input_tensor, filter=weights,
                            strides=strides, padding=padding_mode)
        print(conv)

        # conv = tf.nn.bias_add(value=conv, bias=biases,
        #                       name='net_pre-activation')
        # print(conv)

        return conv


def fc_layer(input_tensor, name, n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))

        weights_shape = [n_input_units, n_output_units]
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        print(weights)

        biases = tf.get_variable(
            name='_biases', initializer=tf.zeros(shape=[n_output_units]))
        print(biases)

        layer = tf.matmul(input_tensor, weights)
        print(layer)

        layer = tf.nn.bias_add(layer, biases, name='net_pre-activation')
        print(layer)

        if activation_fn is None:
            return layer

        layer = activation_fn(layer, name='activation')
        print(layer)

        return layer


def build_generator(learning_rate, height, width, channels, resize, alpha, block_count):
    tf_x = tf.placeholder(dtype=tf.float32, shape=[
                          None, height*width], name='tf_x')
    tf_y = tf.placeholder(dtype=tf.float32, shape=[
                          None, resize*resize*height*width], name='tf_y')

    tf_x_image = tf.reshape(
        tf_x, shape=[-1, height, width, channels], name='tf_x_reshaped')
    tf_y_image = tf.reshape(
        tf_y, shape=[-1, resize*height, resize*width, channels], name='tf_y_reshaped')

    print('\nBuilding first layer:')
    h1 = conv_layer(input_tensor=tf_x_image, name='conv_1',
                    kernel_size=(9, 9), n_output_channels=64,
                    padding_mode='SAME')

    h1_activation = tf.identity(
        prelu(h1, alpha), name='h1_activation')
    print(h1_activation)

    print('\nBuilding block layers:')
    blocks = [residual_block(input_tensor=h1_activation, name='block_0',
                             kernel_size=3, n_output_channels=64, strides=(1, 1, 1, 1))]
    for _ in range(block_count-1):
        blocks.append(residual_block(input_tensor=blocks[-1], name='block_'+len(
            blocks), kernel_size=3, n_output_channels=64, strides=(1, 1, 1, 1)))

    print('\nBuilding pre upscale layer:')
    conv_pre_upscale_1 = conv_layer(blocks[-1], kernel_size=3, name='conv_pre_upscale_1', n_output_channels=64,
                                    padding_mode='SAME')

    conv_pre_upscale_1 = batch_normalize(conv_pre_upscale_1)

    conv_pre_upscale_normalized_1 = tf.add(
        conv_pre_upscale_1, h1_activation, name='conv_pre_upscale_normalized_1')
    print(conv_pre_upscale_normalized_1)

    print('\nBuilding upscale layers:')
    upscale_layers = [upscale_layer(input_tensor=conv_pre_upscale_normalized_1, name='upscale_layer_0',
                                    kernel_size=3, n_output_channels=256, strides=(1, 1, 1, 1))]
    for _ in range(resize-1):
        upscale_layers.append(upscale_layer(input_tensor=blocks[-1], name='upscale_layer_'+len(
            blocks), kernel_size=3, n_output_channels=256, strides=(1, 1, 1, 1)))

    output = conv_layer(input_tensor=upscale_layers[-1], name='output',
                        kernel_size=(9, 9), n_output_channels=3,
                        padding_mode='SAME')

    # predictions = {
    #     'probabilities': tf.nn.softmax(h4, name='probabilities'),
    #     'labels': tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels')
    # }

    mse_loss = tf.losses.mean_squared_error(tf_y_image, output)

    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #     logits=h4, labels=tf_y_onehot), name='cross_entropy_loss')

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(mse_loss, name='train_op')

    # correct_predictions = tf.equal(
    #     predictions['labels'], tf_y, name='correct_predictions')

    # accuracy = tf.reduce_mean(
    #     tf.cast(correct_predictions, tf.float32), name='accuracy')
