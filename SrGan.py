import tensorflow as tf
import numpy as np
import sklearn
import os
from ImageLoader import ImageLoader


class SrGan(object):
    def __init__(self, epochs, learning_rate=0.0001, channels=3, resize=2, alpha=0.2, block_count=16):
        self.learning_rate = learning_rate
        self.channels = channels
        self.resize = resize
        self.alpha = alpha
        self.block_count = block_count
        self.epochs = epochs

        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(1)
            self.build_generator()
            self.init_op = tf.global_variables_initializer()

            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=graph)

    def prelu(self, x, alpha):
        return tf.nn.relu(x)+alpha*(x-tf.abs(x))*0.5

    def upscale_layer(self, input_tensor, name,
                      kernel_size, n_output_channels,
                      padding_mode='SAME', strides=(1, 1, 1, 1), alpha=0.2):

        with tf.variable_scope(name):
            input_shape = input_tensor.get_shape().as_list()
            n_input_channels = input_shape[-1]

            weights_shape = list(kernel_size) + \
                [n_input_channels, n_output_channels]
            weights = tf.get_variable(
                name='_weights', shape=weights_shape, initializer=tf.contrib.layers.xavier_initializer())
            print(weights)

            upscale_layer = tf.nn.conv2d(input=input_tensor, filter=weights,
                                         strides=strides, padding=padding_mode)
            print(upscale_layer)

            upscale_layer = tf.depth_to_space(
                upscale_layer, 2, name='upscale_layer_preactivation')
            print(upscale_layer)

            upscale_layer = tf.identity(
                self.prelu(upscale_layer, alpha), name='activation')
            print(upscale_layer)

            return upscale_layer

    def residual_block(self, input_tensor, name,
                       kernel_size, n_output_channels,
                       padding_mode='SAME', strides=(1, 1, 1, 1), alpha=0.2):

        with tf.variable_scope(name):
            input_shape = input_tensor.get_shape().as_list()
            n_input_channels = input_shape[-1]

            weights_shape = list(kernel_size) + \
                [n_input_channels, n_output_channels]
            weights1 = tf.get_variable(
                name='_weights1', shape=weights_shape, initializer=tf.contrib.layers.xavier_initializer())
            weights2 = tf.get_variable(
                name='_weights2', shape=weights_shape, initializer=tf.contrib.layers.xavier_initializer())

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

            residual_block = self.batch_normalize(residual_block)

            residual_block = tf.identity(
                self.prelu(residual_block, alpha), name='activation')
            print(residual_block)

            residual_block = tf.nn.conv2d(input=residual_block, filter=weights2,
                                          strides=strides, padding=padding_mode)
            print(residual_block)

            # residual_block = tf.nn.bias_add(value=residual_block, bias=biases,
            #                                 name='net_post-activation')
            # print(residual_block)
            residual_block = self.batch_normalize(residual_block)

            residual_block = tf.add(
                residual_block, input_tensor, name='residual_output')
            print(residual_block)

            return residual_block

    def batch_normalize(self, residual_block):
        mean, var = tf.nn.moments(residual_block, axes=[0, 1, 2])
        residual_block = tf.nn.batch_normalization(
            residual_block, mean=mean, variance=var, offset=0, scale=1, variance_epsilon=1e-3, name='normalized_post-activation')
        print(residual_block)
        return residual_block

    def conv_layer(self, input_tensor, name,
                   kernel_size, n_output_channels,
                   padding_mode='SAME', strides=(1, 1, 1, 1)):

        with tf.variable_scope(name):
            input_shape = input_tensor.get_shape().as_list()
            n_input_channels = input_shape[-1]

            weights_shape = list(kernel_size) + \
                [n_input_channels, n_output_channels]
            weights = tf.get_variable(name='_weights', shape=weights_shape,
                                      initializer=tf.contrib.layers.xavier_initializer())

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

    def fc_layer(self, input_tensor, name, n_output_units, activation_fn=None):
        with tf.variable_scope(name):
            input_shape = input_tensor.get_shape().as_list()[1:]
            n_input_units = np.prod(input_shape)
            if len(input_shape) > 1:
                input_tensor = tf.reshape(
                    input_tensor, shape=(-1, n_input_units))

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

    def build_generator(self):
        tf_x_image = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, self.channels], name='tf_x')
        tf_y_image = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, self.channels], name='tf_y')

        print('\nBuilding first layer:')
        h1 = self.conv_layer(input_tensor=tf_x_image, name='conv_1',
                             kernel_size=(9, 9), n_output_channels=64,
                             padding_mode='SAME')

        h1_activation = tf.identity(
            self.prelu(h1, self.alpha), name='h1_activation')
        print(h1_activation)

        print('\nBuilding block layers:')
        blocks = [self.residual_block(input_tensor=h1_activation, name='block_0',
                                      kernel_size=(3, 3), n_output_channels=64, strides=(1, 1, 1, 1))]
        for i in range(self.block_count-1):
            blocks.append(self.residual_block(input_tensor=blocks[-1], name='block_'+str(
                i+1), kernel_size=(3, 3), n_output_channels=64, strides=(1, 1, 1, 1)))

        print('\nBuilding pre upscale layer:')
        conv_pre_upscale_1 = self.conv_layer(blocks[-1], kernel_size=(3, 3), name='conv_pre_upscale_1', n_output_channels=64,
                                             padding_mode='SAME')

        conv_pre_upscale_1 = self.batch_normalize(conv_pre_upscale_1)

        conv_pre_upscale_normalized_1 = tf.add(
            conv_pre_upscale_1, h1_activation, name='conv_pre_upscale_normalized_1')
        print(conv_pre_upscale_normalized_1)

        print('\nBuilding upscale layers:')
        upscale_layers = [self.upscale_layer(input_tensor=conv_pre_upscale_normalized_1, name='upscale_layer_0',
                                             kernel_size=(3, 3), n_output_channels=256, strides=(1, 1, 1, 1))]
        for i in range(self.resize-1):
            upscale_layers.append(self.upscale_layer(input_tensor=upscale_layers[-1], name='upscale_layer_'+str(
                i+1), kernel_size=(3, 3), n_output_channels=256, strides=(1, 1, 1, 1)))

        output = self.conv_layer(input_tensor=upscale_layers[-1], name='output',
                                 kernel_size=(9, 9), n_output_channels=3,
                                 padding_mode='SAME')

        output = tf.identity(output, name='output_image')
        print(output)
        # predictions = {
        #     'probabilities': tf.nn.softmax(h4, name='probabilities'),
        #     'labels': tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels')
        # }

        mse_loss = tf.losses.mean_squared_error(tf_y_image, output)
        mse_loss = tf.identity(mse_loss, name='mse_loss')
        # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     logits=h4, labels=tf_y_onehot), name='cross_entropy_loss')

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer = optimizer.minimize(mse_loss, name='train_op')

        # correct_predictions = tf.equal(
        #     predictions['labels'], tf_y, name='correct_predictions')

        # accuracy = tf.reduce_mean(
        #     tf.cast(correct_predictions, tf.float32), name='accuracy')

    def train(self, preload_epoch=0, validation_set=None, initialize=True):

        training_loss = []

        if initialize:
            self.sess.run(self.init_op)
        image_loader = ImageLoader(batch_size=8)
        for epoch in range(1, self.epochs+1):
            print('Epoch '+str(epoch))
            image_loader.shuffle_data()
            batch_gen = image_loader.getImages()

            avg_loss = 0.0
            for i, (batch_x, batch_y) in enumerate(batch_gen):
                if(i % 500 == 0):
                    print('    batch ' + str(i)+'/' +
                          str(image_loader.batch_count))
                feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y}

                loss, _ = self.sess.run(
                    ['mse_loss:0', 'train_op'], feed_dict=feed)
                avg_loss += loss

            training_loss.append(avg_loss/(i+1))
            print('Epoch %02d Training Avg. Loss: %7.3f' %
                  (epoch, avg_loss), end=' ')
            self.save(epoch=preload_epoch+epoch)

            if validation_set is not None:
                feed = {'tf_x:0': validation_set[0], 'tf_y:0': validation_set[1],
                        'is_train:0': False}
                validation_acc = self.sess.run('accuracy:0', feed_dict=feed)
                print(' Validation Acc: %7.3f' % validation_acc)
            else:
                print()

    # def create_batch_generator(self, X, y, batch_size=128, shuffle=False, random_seed=None):
    #     if shuffle:
    #         X_copy, y_copy = sklearn.utils.shuffle(X, y)

    #     for i in range(0, X.shape[0], batch_size):
    #         yield (X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])

    def save(self, epoch, path='./mse-model/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        print('Saving model in %s' % path)
        self.saver.save(self.sess, os.path.join(path, 'model.ckpt'),
                        global_step=epoch)

    def load(self, path, epoch):
        print('Loading model from %s' % path)
        self.saver.restore(self.sess, os.path.join(
            path, 'model.ckpt-%d' % epoch))

    def predict(self,  X_test):
        feed = {'tf_x:0': X_test}
        return self.sess.run('output_image:0', feed_dict=feed)
