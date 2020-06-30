import tensorflow as tf
from functools import partial
import numpy as np
import tensorflow.contrib.slim as slim


def fc(inputs, num_outputs, **kwargs):
    with tf.variable_scope(None, 'flatten_fully_connected', [inputs]):
        if inputs.shape.ndims > 2:
            inputs = slim.flatten(inputs)
        return slim.fully_connected(inputs, num_outputs, activation_fn=None, **kwargs)


conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)


class Model:
    def __init__(self, img, z_dim, lr, model):
        img_shape = img.shape[1:]
        self.z_dim = z_dim
        self.channels = img_shape[2]
        
        # input
        z_sample = tf.placeholder(tf.float32, [None, z_dim])
        
        # encode & decode
        z_mu, z_log_sigma_sq, img_rec = self.enc_dec(img)
        
        # loss
        if model == 'mse':
            rec_loss = tf.losses.mean_squared_error(img, img_rec)
            kld_loss = -tf.reduce_mean(0.5 * (1 + z_log_sigma_sq - z_mu ** 2 - tf.exp(z_log_sigma_sq)))
        else:
            if model == 'gaussian':
                log_sigma = tf.Variable(0.0, trainable=False)
            elif model == 'sigma':
                log_sigma = tf.Variable(0.0, trainable=True)

            rec_loss = tf.reduce_sum(gaussian_nll(img_rec, log_sigma, img))
            kld_loss = -tf.reduce_sum(0.5 * (1 + z_log_sigma_sq - z_mu ** 2 - tf.exp(z_log_sigma_sq)))
        
        loss = rec_loss + kld_loss
        
        # optim
        step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss / np.sum(img.shape).value)
        
        # summary
        summary = [tf.summary.scalar('rec_loss', rec_loss), tf.summary.scalar('kld_loss', kld_loss)]
        
        # sample
        _, _, img_rec_sample = self.enc_dec(img, is_training=False)
        img_sample = self.dec(z_sample, is_training=False)
        
        self.params = summary, step, img_sample, z_sample, img_rec_sample
        
    def get_params(self):
        return self.params
    
    def enc_dec(self, img, is_training=True):
        # encode
        z_mu, z_log_sigma_sq = self.enc(img, is_training=is_training)
        
        # sample
        epsilon = tf.random_normal(tf.shape(z_mu))
        if is_training:
            z = z_mu + tf.exp(0.5 * z_log_sigma_sq) * epsilon
        else:
            z = z_mu
        
        # decode
        img_rec = self.dec(z, is_training=is_training)
        
        return z_mu, z_log_sigma_sq, img_rec

    def enc(self, img, dim=64, is_training=True):
        bn = partial(batch_norm, is_training=is_training)
        conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)
    
        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            y = conv_bn_lrelu(img, dim, 5, 2)
            y = conv_bn_lrelu(y, dim * 2, 5, 2)
            z_mu = fc(y, self.z_dim)
            z_log_sigma_sq = fc(y, self.z_dim)
            return z_mu, z_log_sigma_sq

    def dec(self, z, dim=64, is_training=True):
        bn = partial(batch_norm, is_training=is_training)
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)
    
        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = relu(fc(z, 7 * 7 * dim * 2))
            y = tf.reshape(y, [-1, 7, 7, dim * 2])
            y = dconv_bn_relu(y, dim * 1, 5, 2)
            img = tf.tanh(dconv(y, self.channels, 5, 2))
            return img


def gaussian_nll(mu, log_sigma, x):
    return 0.5 * ((x - mu) / tf.exp(log_sigma)) ** 2 + log_sigma + 0.5 * np.log(2 * np.pi)
