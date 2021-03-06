import tensorflow as tf


def square_root_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
    r"""Performs square root pooling on the input.

    Each entry in `output` is the mean of the corresponding size `ksize`

    window in `value`.

    Args:
      value: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
        4-D with shape `[batch, height, width, channels]`.
      ksize: A list of `ints` that has length `>= 4`.
        The size of the sliding window for each dimension of `value`.
      strides: A list of `ints` that has length `>= 4`.
        The stride of the sliding window for each dimension of `value`.
      padding: A `string` from: `"SAME", "VALID"`.
        The type of padding algorithm to use.
      data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
        Specify the data format of the input and output data. With the

        default format "NHWC", the data is stored in the order of:

            [batch, in_height, in_width, in_channels].

        Alternatively, the format could be "NCHW", the data storage order of:

            [batch, in_channels, in_height, in_width].
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `value`.
    """
    value = tf.square(value)
    inshape = value.get_shape().as_list()
    shape = [ksize[1], ksize[2], inshape[3], inshape[3]]
    # print(inshape, "==============================", shape)
    filters = tf.ones(shape=shape, dtype=value.dtype)
    # return tf.sqrt(tf.nn.avg_pool(value, ksize, strides, padding, data_format="NHWC", name=None))
    return tf.sqrt(tf.nn.conv2d(value, filters, strides, padding=padding, data_format='NHWC', dilations=[1, 1, 1, 1], name=name))





