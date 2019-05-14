#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-19 23:33:29
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import os
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import tensorflow as tf
import tftool


def plot_conv_weights(weights, plot_dir, name, channels_all=True, filters_all=True, channels=[0], filters=[0], cmap='seismic'):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """

    w_min = np.min(weights)
    w_max = np.max(weights)

    # print("++++++++++++++++++", weights.shape)
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    if filters_all:
        num_filters = weights.shape[3]
        filters = range(weights.shape[3])
    else:
        num_filters = len(filters)

    # get number of grid rows and columns
    grid_r, grid_c = tftool.get_grid_dim(num_filters)

    # create figure and axes

    # print(channels, "==================")
    # iterate channels
    for channel_ID in channels:
        print("  channel :", channel_ID)
        fig, axes = plt.subplots(min([grid_r, grid_c]),
                                 max([grid_r, grid_c]))
        # print(channel_ID)
        # iterate filters inside every channel
        if num_filters == 1:
            img = weights[:, :, channel_ID, filters[0]]
            axes.imshow(img, vmin=w_min, vmax=w_max,
                        interpolation='nearest', cmap=cmap)
            # remove any labels from the axes
            axes.set_xticks([])
            axes.set_yticks([])
        else:
            for l, ax in enumerate(axes.flat):
                # print(channel_ID, l)
                # get a single filter
                # print(channel_ID, filters[l], "+++++++++++++++")
                img = weights[:, :, channel_ID, filters[l]]
                # put it on the grid
                ax.imshow(img, vmin=w_min, vmax=w_max,
                          interpolation='nearest', cmap=cmap)
                # remove any labels from the axes
                ax.set_xticks([])
                ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(
            plot_dir, '{}-chl{}.png'.format(name, channel_ID)),
            bbox_inches='tight')
        plt.close()


if __name__ == '__main__':

    visuallayers = ['conv1_1', 'conv1_2',
                    'conv2_1', 'conv2_2',
                    'conv3_1', 'conv3_2']

    model_path = ''
    image_path = 'images/train_images/sunny_0058.jpg'

    dir_prefix = ''
    n_input = 784  # MNIST data input (img shape: 28*28)
    n_classes = 10  # MNIST total classes (0-9 digits)
    dropout = 0.75  # Dropout, probability to keep units

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    with tf.Session(graph=tf.get_default_graph()) as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, model_path)

        img = imread(image_path)
        img = np.float32(img)
        img = np.expand_dims(img, axis=0)

        conv_out = sess.run(tf.get_collection('activations'),
                            feed_dict={x: img, keep_prob: 1.0})
        for i, layer in enumerate(visuallayers):
            tftool.create_dir(dir_prefix + layer)
            for j in range(conv_out[i].shape[3]):
                tftool.plot_conv_output(
                    conv_out[i], dir_prefix + layer, str(j),
                    filters_all=False, filters=[j])

        sess.close()


