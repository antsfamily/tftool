#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-31 23:23:29
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import os
import tensorflow as tf


def load_ckpt(sess, ckpt_dir, step):

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join(ckpt_dir, oldname)
    newname = os.path.join(ckpt_dir, newname)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt-' + str(step)
    filename = os.path.join(ckpt_dir, filename)
    saver.restore(sess, filename)

    return sess

    print("Checkpoint of step " + str(step) + " restored!")
