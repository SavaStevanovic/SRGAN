import tensorflow as tf
import numpy as np
import sklearn
import os
from ImageLoader import ImageLoader


class SrGan(object):
    def __init__(self, epochs, learning_rate=0.00001, channels=3, resize=2, alpha=0.2, block_count=16):
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

    def build_generator(self):
        tf_x_image = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, self.channels], name='tf_x')
        tf_y_image = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, self.channels], name='tf_y')
        tf_training = tf.placeholder(
            dtype=tf.bool, shape=None, name='tf_training')

        print('\nBuilding first layer:')
        net = tf.layers.conv2d(inputs=tf_x_image, filters=64, kernel_size=(
            9, 9), activation=tf.nn.leaky_relu, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        net_pre = net
        post_res = net_pre

        print('\nBuilding block layers:')
        for i in range(self.block_count):
            net = tf.layers.conv2d(
                inputs=net, filters=64, kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.batch_normalization(net, training=tf_training)
            net = tf.nn.leaky_relu(net)
            net = tf.layers.conv2d(
                inputs=net, filters=64, kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.batch_normalization(net, training=tf_training)
            net += net_pre
            net_pre = net

        print('\nBuilding pre upscale layer:')
        net = tf.layers.conv2d(inputs=net, filters=64,
                               kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        net = tf.layers.batch_normalization(net, training=tf_training)
        net += post_res

        print('\nBuilding upscale layers:')
        for i in range(self.resize):
            net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(
                3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.depth_to_space(net, block_size=2)
            net = tf.nn.leaky_relu(net)

        output = tf.layers.conv2d(
            inputs=net, filters=3, kernel_size=(9, 9), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.tanh)
        output = tf.identity(output, name='output_image')

        mse_loss = tf.losses.mean_squared_error(tf_y_image, output)
        mse_loss = tf.identity(mse_loss, name='mse_loss')

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer = optimizer.minimize(mse_loss, name='train_op')

    def train(self, preload_epoch=0, validation_set_path=None, initialize=True):
        writer = tf.summary.FileWriter(logdir='./logs/', graph=self.sess.graph)
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
                if(i % 1000 == 0):
                    print('    batch ' + str(i)+'/' +
                          str(image_loader.batch_count))
                feed = {'tf_x:0': batch_x,
                        'tf_y:0': batch_y, 'tf_training:0': True}

                loss, _ = self.sess.run(
                    ['mse_loss:0', 'train_op'], feed_dict=feed)
                avg_loss += loss

            training_loss.append(avg_loss/(i+1))
            print('Epoch %02d Training Avg. Loss: %7.10f' %
                  (preload_epoch+epoch, avg_loss/(i+1)), end=' ')
            self.save(epoch=preload_epoch+epoch)

            if validation_set_path is not None:
                val_image_loader = ImageLoader(
                    batch_size=8, image_dir=validation_set_path)
                val_batch_gen = val_image_loader.getImages()
                val_avg_loss = 0.0
                for i, (batch_x, batch_y) in enumerate(val_batch_gen):
                    if(i % 200 == 0):
                        print('    validation batch ' + str(i)+'/' +
                              str(val_image_loader.batch_count))
                    feed = {'tf_x:0': batch_x,
                            'tf_y:0': batch_y, 'tf_training:0': False}

                    val_loss = self.sess.run(
                        'mse_loss:0', feed_dict=feed)
                    val_avg_loss += val_loss

                print('Epoch %02d Validation Avg. Loss: %7.10f' %
                      (preload_epoch+epoch, val_avg_loss/(i+1)))
            else:
                print()
        writer.close()

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
        feed = {'tf_x:0': X_test, 'tf_training:0': True}
        return self.sess.run('output_image:0', feed_dict=feed)
