from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import os
import tensorflow as tf
import imageio

import utils
from model import Model


## Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', dest='epoch', type=int, default=10)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model', type=str, default='mse', help='which model to use: mse, gaussian or sigma')
parser.add_argument('--experiment_name', dest='experiment_name', default='exp')
args = parser.parse_args()

epoch = args.epoch
batch_size = args.batch_size
lr = args.lr
experiment_name = args.experiment_name
model = args.model
z_dim = 20

## Logging
os.makedirs('./output/%s' % experiment_name, exist_ok=True)


## Dataset
ds_train, ds_length = utils.get_svhn()
# ds_train, ds_length = datasets.get_random_data()
ds_train = ds_train.map(lambda x: tf.image.resize_images(tf.cast(x, tf.float32) / 127.5 - 1, (28, 28)))
ds_train = ds_train.shuffle(1000).batch(batch_size, drop_remainder=True).prefetch(10)
ds_length = ds_length // batch_size

train_iter = ds_train.make_initializable_iterator()
img = train_iter.get_next()
img_shape = img.shape[1:]


## Build Graph
model = Model(img, z_dim, lr, model)
summary, step, img_sample, z_sample, img_rec_sample = model.get_params()


## Init
sess = tf.Session()

summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)
sess.run(tf.global_variables_initializer())


## Train
try:
    z_ipt_sample = np.random.normal(size=[100, z_dim])
    it = -1
    
    for ep in range(epoch):
        it_per_epoch = it_in_epoch if it != -1 else -1
        it_in_epoch = 0
        # Reinit
        sess.run(train_iter.initializer)
        
        # The last two elements in the dataset are used for visualization, so I crop the dataset just in case
        # TODO: fix this
        for batch in range(ds_length - 10):
            it += 1
            it_in_epoch += 1

            # train
            summary_opt, _ = sess.run([summary, step])
            for summary_opt_i in summary_opt: summary_writer.add_summary(summary_opt_i, it)

            # display
            if (it + 1) % 100 == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (ep, it_in_epoch, it_per_epoch))

            # sample
            if (it + 1) % 1000 == 0:
                save_dir = './output/%s/sample_training' % experiment_name
                os.makedirs(save_dir, exist_ok=True)

                img_rec_opt_sample, img_data = sess.run([img_rec_sample, img])
                ipt_rec = np.concatenate((img_data, img_rec_opt_sample), axis=2).squeeze()
                img_opt_sample = sess.run(img_sample, feed_dict={z_sample: z_ipt_sample}).squeeze()

                # TODO remove immerge
                imageio.imwrite('%s/Epoch_(%d)_img_rec.jpg' % (save_dir, ep), utils.immerge(ipt_rec) / 2 + 0.5)

                imageio.imwrite('%s/Epoch_(%d)_img_sample.jpg' % (save_dir, ep), utils.immerge(img_opt_sample) / 2 + 0.5)
finally:
    sess.close()
