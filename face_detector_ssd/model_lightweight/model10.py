import tensorflow as tf

NAME = 'model10_lw'

IMAGE_SIZE = 128
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
  box_layer = tf.layers.conv2d(input, OFFSET + CLASSES, kernel_shape, strides, padding='SAME')
  shape = tf.shape(box_layer)
  return tf.reshape(box_layer, [shape[0], shape[1] * shape[2], shape[3]])


def ssd_layers(feature_map, is_train=True):
  with tf.variable_scope('%s_ssd_layers' % NAME):
    outputs = []

    conv = feature_map

    for index, filters in enumerate([192, 128, 64]):
      with tf.variable_scope('ssd_block_%d' % index):
        conv_shape = conv.get_shape()
        print(conv_shape)

        outputs.append(_create_box_layer(conv, [3, 3], [1, 1]))

        conv = tf.layers.conv2d(conv, filters, [3, 3], [2, 2],
                                padding='SAME',
                                activation=tf.nn.relu,
                                use_bias=True,
                                trainable=is_train,
                                name='pooling')

    return tf.concat(outputs, axis=1)
