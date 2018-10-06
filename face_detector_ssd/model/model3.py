import tensorflow as tf

NAME = 'model3'

IMAGE_SIZE = 256
CHANNELS = 3

CLASSES = 1
OFFSET = 4


def base_layers(image, is_train=True):
  with tf.variable_scope(NAME):
    conv = tf.layers.conv2d(image, 64, [3, 3], [1, 1],
                            padding='SAME',
                            activation=tf.nn.relu,
                            use_bias=True,
                            trainable=is_train)
    conv = tf.layers.conv2d(conv, 64, [3, 3], [1, 1],
                            padding='SAME',
                            activation=tf.nn.relu,
                            use_bias=True,
                            trainable=is_train)
    pool = tf.layers.max_pooling2d(conv, [3, 3], [2, 2], padding='SAME')

    conv = tf.layers.conv2d(pool, 128, [3, 3], [1, 1],
                            padding='SAME',
                            activation=tf.nn.relu,
                            use_bias=True,
                            trainable=is_train)
    conv = tf.layers.conv2d(conv, 128, [3, 3], [1, 1],
                            padding='SAME',
                            activation=tf.nn.relu,
                            use_bias=True,
                            trainable=is_train)
    pool = tf.layers.max_pooling2d(conv, [3, 3], [2, 2], padding='SAME')

    conv = tf.layers.conv2d(pool, 256, [3, 3], [1, 1],
                            padding='SAME',
                            activation=tf.nn.relu,
                            use_bias=True,
                            trainable=is_train)
    conv = tf.layers.conv2d(conv, 256, [3, 3], [1, 1],
                            padding='SAME',
                            activation=tf.nn.relu,
                            use_bias=True,
                            trainable=is_train)
    pool = tf.layers.max_pooling2d(conv, [3, 3], [2, 2], padding='SAME')

    return pool


def _create_box_layer(input, kernel_shape, strides):
  offset = tf.layers.conv2d(input, OFFSET,
                            kernel_shape, strides, padding='SAME')
  classes = tf.layers.conv2d(input, CLASSES,
                             kernel_shape, strides, padding='SAME')
  temp = tf.concat([offset, classes], axis=3)
  shape = tf.shape(temp)
  return tf.reshape(temp, [shape[0], shape[1] * shape[2], shape[3]])


def ssd_layers(base_logits, is_train=True):
  outputs = []

  conv = base_logits

  for index in range(4):
    with tf.variable_scope('ssd_block_%d' % index):

      conv = tf.layers.conv2d(conv, 256, [3, 3], [2, 2],
                              padding='SAME',
                              activation=tf.nn.relu,
                              use_bias=True,
                              trainable=is_train,
                              name='pooling')

      conv_shape = conv.get_shape()
      print(conv_shape)

      if conv_shape[1].value == 1:
        continue

      outputs.append(_create_box_layer(conv, [2, 2], [1, 1]))
      # outputs.append(_create_box_layer(conv, [2, 1], [1, 1]))
      # outputs.append(_create_box_layer(conv, [1, 2], [1, 1]))

      if conv_shape[1].value <= 2:
        continue

      outputs.append(_create_box_layer(conv, [3, 2], [1, 1]))
      outputs.append(_create_box_layer(conv, [2, 3], [1, 1]))
      # outputs.append(_create_box_layer(conv, [3, 3], [1, 1]))

  return tf.concat(outputs, axis=1)
