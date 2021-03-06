import sys

import tensorflow as tf

from playground.common import create_ssd_layer, create_boxes


def create_ssd_layers(base_logits):
  ssd_layers = []

  conv = base_logits

  for index in range(5):
    conv_shape = conv.get_shape()

    if conv_shape[1].value <= 2:
      continue

    ssd_layers.append(create_ssd_layer(conv, [3, 3], [1, 1]))
    # ssd_layers.append(create_ssd_layer(conv, [2, 1], [1, 1]))
    # ssd_layers.append(create_ssd_layer(conv, [1, 2], [1, 1]))

    if conv_shape[1].value <= 5:
      continue

    ssd_layers.append(create_ssd_layer(conv, [5, 3], [1, 1]))
    ssd_layers.append(create_ssd_layer(conv, [3, 5], [1, 1]))
    # ssd_layers.append(create_ssd_layer(conv, [3, 3], [1, 1]))

    conv = tf.layers.conv2d(conv, 5, [3, 3], [2, 2], padding='SAME')

  return ssd_layers


INPUT_SHAPE = [1, 32, 32, 1]


def main(argv=None):
  feature_map = tf.get_variable(shape=INPUT_SHAPE,
                                name='feature_map')

  ssd_layers = create_ssd_layers(feature_map)
  boxes = create_boxes(ssd_layers)

  print('box count %d' % len(boxes))
  for box in boxes:
    print(box)


if __name__ == "__main__":
  main(sys.argv)
