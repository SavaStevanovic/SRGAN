import tensorflow as tf
import numpy as np
import sklearn
import os
from ImageLoader import ImageLoader
# from tensorflow.contrib.keras.api.keras.applications import VGG19
# from tensorflow.contrib.keras.api.keras.models import Model
import tensorlayer as tl
import vgg19


class SrGan(object):
    def __init__(self, epochs, learning_rate=0.000001, channels=3, resize=2, alpha=0.2, block_count=16):
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

    def build_discriminator(self, tf_image, tf_training, reuse):
        with tf.variable_scope("discriminator", reuse=reuse) as dis:
            print('Building descriminator')

            with tf.variable_scope("discriminator_64_1"):
                net = tf.layers.conv2d(inputs=tf_image, filters=64, kernel_size=(
                    3, 3), activation=tf.nn.leaky_relu, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())

            with tf.variable_scope("discriminator_64_2"):
                net = tf.layers.conv2d(
                    inputs=net, filters=64, kernel_size=(3, 3), padding='SAME', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.layers.batch_normalization(net, training=tf_training)
                net = tf.nn.leaky_relu(net)

            with tf.variable_scope("discriminator_128_1"):
                net = tf.layers.conv2d(
                    inputs=net, filters=128, kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.layers.batch_normalization(net, training=tf_training)
                net = tf.nn.leaky_relu(net)

            with tf.variable_scope("discriminator_128_2"):
                net = tf.layers.conv2d(
                    inputs=net, filters=128, kernel_size=(3, 3), padding='SAME', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.layers.batch_normalization(net, training=tf_training)
                net = tf.nn.leaky_relu(net)

            with tf.variable_scope("discriminator_256_1"):
                net = tf.layers.conv2d(
                    inputs=net, filters=256, kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.layers.batch_normalization(net, training=tf_training)
                net = tf.nn.leaky_relu(net)

            with tf.variable_scope("discriminator_256_2"):
                net = tf.layers.conv2d(
                    inputs=net, filters=256, kernel_size=(3, 3), padding='SAME', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.layers.batch_normalization(net, training=tf_training)
                net = tf.nn.leaky_relu(net)

            with tf.variable_scope("discriminator_512_1"):
                net = tf.layers.conv2d(
                    inputs=net, filters=512, kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.layers.batch_normalization(net, training=tf_training)
                net = tf.nn.leaky_relu(net)

            with tf.variable_scope("discriminator_512_2"):
                net = tf.layers.conv2d(
                    inputs=net, filters=512, kernel_size=(3, 3), padding='SAME', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.layers.batch_normalization(net, training=tf_training)
                net = tf.nn.leaky_relu(net)

            with tf.variable_scope("flatten"):
                net = tf.reshape(net, shape=[-1, 512*6*6])

            with tf.variable_scope("dense_1"):
                net = tf.layers.dense(inputs=net, units=1024,
                                    activation=tf.nn.leaky_relu)
            with tf.variable_scope("dense_2"):
                net = tf.layers.dense(inputs=net, units=1)
            return net

    def build_generator(self):
        tf_x_image = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, self.channels], name='tf_x')
        tf_y_image = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, self.channels], name='tf_y')
        tf_training = tf.placeholder(
            dtype=tf.bool, shape=None, name='tf_training')
        with tf.variable_scope("srgan") as vs:
            print('\nBuilding first layer:')
            net = tf.layers.conv2d(inputs=tf_x_image, filters=64, kernel_size=(
                9, 9), activation=tf.nn.leaky_relu, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_pre = net
            post_res = net_pre

            print('\nBuilding block layers:')
            with tf.variable_scope("residual_blocks"):
                for i in range(self.block_count):
                    with tf.variable_scope("residual_block_"+str(i)):
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
                with tf.variable_scope("upscale_layer_"+str(i)):
                    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(
                        3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                    net = tf.depth_to_space(net, block_size=2)
                    net = tf.nn.leaky_relu(net)

            output = tf.layers.conv2d(
                inputs=net, filters=3, kernel_size=(9, 9), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.tanh)
            output = tf.identity(output, name='output_image')

        output224 = tf.image.resize_images(
            output, size=[224, 224], method=0, align_corners=False)
        tf_y_image244 = tf.image.resize_images(
            tf_y_image, size=[224, 224], method=0, align_corners=False)

        self.vgg, output_content = vgg19.Vgg19_simple_api(
            (output224+1)/2, reuse=False)
        _, target_content = vgg19.Vgg19_simple_api(
            (tf_y_image244+1)/2, reuse=True)

        discriminator_logits_gen = self.build_discriminator(
            output, tf_training, reuse=False)
        discriminator_logits_real = self.build_discriminator(
            tf_y_image, tf_training, reuse=True)

        discriminator_loss = tl.cost.sigmoid_cross_entropy(discriminator_logits_gen, tf.zeros_like(
            discriminator_logits_gen)) + tl.cost.sigmoid_cross_entropy(discriminator_logits_real, tf.ones_like(discriminator_logits_real))
        discriminator_loss_summ = tf.summary.scalar(
            tensor=discriminator_loss, name='discriminator_loss_summ')

        gen_loss = 0.001 * \
            tl.cost.sigmoid_cross_entropy(
                discriminator_logits_gen, tf.ones_like(discriminator_logits_gen))
        gen_loss_summ = tf.summary.scalar(
            tensor=gen_loss, name='gen_loss_summ')

        content_loss = 0.006*tl.cost.mean_squared_error(
            target_content.outputs, output_content.outputs, is_mean=True)
        content_loss_summ = tf.summary.scalar(
            tensor=content_loss, name='content_loss_summ')

        mse_loss = tl.cost.mean_squared_error(
            tf_y_image, output, is_mean=True)
        self.mse_loss_summ = tf.summary.scalar(
            tensor=mse_loss, name='mse_loss_summ')
        total_loss_summ = tf.summary.scalar(
            tensor=mse_loss+content_loss+gen_loss, name='total_loss_summ')

        srgan_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='srgan')
        pre_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        pre_optimizer = pre_optimizer.minimize(mse_loss, name='train_mse_op', var_list=srgan_variables)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer = optimizer.minimize(
            mse_loss+content_loss+gen_loss, name='train_op', var_list=srgan_variables)
        
        disc_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        disc_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        disc_optimizer = disc_optimizer.minimize(
            discriminator_loss, name='train_op_disc', var_list=disc_variables)
        self.merged = tf.summary.merge_all()

    def train(self, preload_epoch=0, training_path="./ImageNet/TrainImages", validation_set_path="./ImageNet/TestImages", initialize=True, pretrain=False):
        writer = tf.summary.FileWriter(logdir='./logs/', graph=self.sess.graph)
        if initialize:
            self.sess.run(self.init_op)
        self.load_vgg19()
        for epoch in range(1, self.epochs+1):
            self.run_model(writer, training_path, preload_epoch,
                           epoch, training=True, shuffle=True, pretrain=pretrain)

            if validation_set_path is not None:
                self.run_model(writer, validation_set_path,
                               preload_epoch, epoch, training=False)
            else:
                print()
        writer.close()

    def run_model(self, writer, set_path, preload_epoch, epoch, training=False, shuffle=False, pretrain=False):
        if not training:
            print('VALIDATION')
        image_loader = ImageLoader(
            batch_size=8, image_dir=set_path)
        if shuffle:
            image_loader.shuffle_data()
        batch_gen = image_loader.getImages()
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            subbatch = 100
            if(i % subbatch == 0):
                print('batch ' + str(i)+'/'+str(image_loader.batch_count))
                if(training):
                    self.save(epoch=preload_epoch+epoch)
            feed = {'tf_x:0': batch_x,
                    'tf_y:0': batch_y, 'tf_training:0': training}
            if(not training):
                loss = self.sess.run(
                    self.merged, feed_dict=feed)
            else:
                if(not pretrain):
                    loss, _ = self.sess.run(
                        [self.merged, 'train_op_disc'], feed_dict=feed)
                    writer.add_summary(loss)
                    loss, _ = self.sess.run(
                        [self.merged, 'train_op'], feed_dict=feed)
                    writer.add_summary(loss)
                else:
                    loss, _ = self.sess.run(
                        [self.mse_loss_summ, 'train_mse_op'], feed_dict=feed)
                    writer.add_summary(loss)
                # print('Loss: mse-%7.10f , con-%7.10f' %
                #       (loss, content_loss))

    def save(self, epoch, path='./mse-vgg-gen-model/'):
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
        return self.sess.run('srgan/output_image:0', feed_dict=feed)

    def load_vgg19(self):
        vgg19_npy_path = "vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print(
                "Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(vgg19_npy_path, encoding='latin1').item()

        params = []
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        tl.files.assign_params(self.sess, params, self.vgg)
