#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-31 23:23:29
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import os
import tensorflow as tf


def save_ckpt(sess, ckpt_dir, step):

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join(ckpt_dir, oldname)
    newname = os.path.join(ckpt_dir, newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver(sharded=True)
    saver.save(sess, newname, global_step=step)

    print("Checkpoint of step " + str(step) + " saved!")


if __name__ == '__main__':
    pass
