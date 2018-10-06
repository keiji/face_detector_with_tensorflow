import tensorflow as tf

NAME = 'model2'

IMAGE_SIZE = 128
CHANNELS = 3

CLASSES = 1
OFFSET = 4

BASE_CHANNEL = 64
RESIDUAL_CHANNELS = [64, 128, 256]


def base_layers(image, is_train=True):
  with tf.variable_scope(NAME):
    conv = tf.layers.conv2d(image, BASE_CHANNEL, [1, 1], [1, 1],
                            padding='SAME',
                            activation=tf.nn.relu,
                            use_bias=True,
                            trainable=is_train,
                            name='conv_top')

    for index, channel in enumerate(RESIDUAL_CHANNELS):
      with tf.variable_scope('base_residual_block_%d' % index):
        residual = tf.layers.batch_normalization(conv,
                                                 training=is_train,
                                                 trainable=is_train,
                                                 name='bn1')
        residual = tf.nn.relu(residual)

        residual = tf.layers.conv2d(residual, channel, [3, 3], [1, 1],
                                    padding='SAME',
                                    activation=None,
                                    use_bias=False,
                                    trainable=is_train,
                                    name='conv1')

        residual = tf.layers.batch_normalization(residual,
                                                 training=is_train,
                                                 trainable=is_train,
                                                 name='bn2')
        residual = tf.nn.relu(residual)

        residual = tf.layers.conv2d(residual, BASE_CHANNEL, [1, 1], [1, 1],
                                    padding='SAME',
                                    activation=None,
                                    use_bias=False,
                                    trainable=is_train,
                                    name='bottleneck')

        residual = conv + residual
        residual = tf.nn.relu(residual)

        conv = tf.layers.conv2d(residual, BASE_CHANNEL, [3, 3], [2, 2],
                                padding='SAME',
                                activation=tf.nn.relu,
                                use_bias=True,
                                trainable=is_train,
                                name='pooling')
    return conv


def _create_box_layer(input, kernel_shape, strides, is_train):
  temp = tf.layers.conv2d(input,
                          (CLASSES + OFFSET),
                          kernel_shape, strides,
                          padding='SAME',
                          trainable=is_train)
  shape = tf.shape(temp)
  return tf.reshape(temp, [shape[0], shape[1] * shape[2], shape[3]])


def ssd_layers(base_logits, is_train=True):
  with tf.variable_scope('%s_ssd_layers' % NAME):
    outputs = []

    conv = base_logits

    for index in range(4):
      with tf.variable_scope('ssd_block_%d' % index):

        conv_shape = conv.get_shape()
        print(conv_shape)

        if conv_shape[1].value == 1:
          continue

        outputs.append(_create_box_layer(conv, [2, 2], [1, 1], is_train))
        # outputs.append(_create_box_layer(conv, [2, 1], [1, 1], is_train))
        # outputs.append(_create_box_layer(conv, [1, 2], [1, 1], is_train))

        if conv_shape[1].value <= 2:
          continue

        outputs.append(_create_box_layer(conv, [3, 2], [1, 1], is_train))
        outputs.append(_create_box_layer(conv, [2, 3], [1, 1], is_train))
        # outputs.append(_create_box_layer(conv, [3, 3], [1, 1], is_train))

        conv = tf.layers.conv2d(conv, 256, [3, 3], [2, 2],
                                padding='SAME',
                                activation=tf.nn.relu,
                                use_bias=True,
                                trainable=is_train,
                                name='pooling')

    output = tf.concat(outputs, axis=1)
    return output
