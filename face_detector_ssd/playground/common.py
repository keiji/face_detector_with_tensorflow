import tensorflow as tf
import numpy as np

from box import Box


class SsdLayer:
  def __init__(self):
    self.input_shape = None
    self.output_shape = None
    self.kernel_shape = None


def create_ssd_layer(input, kernel_shape, strides):
  temp = tf.layers.conv2d(input, 1, kernel_shape, strides, padding='SAME')

  layer = SsdLayer()
  layer.input_shape = list(map(lambda s: s.value, input.get_shape()))
  layer.kernel_shape = kernel_shape
  layer.output_shape = list(map(lambda s: s.value, temp.get_shape()))

  return layer


def create_boxes(ssd_layers):
  boxes = []

  for index, layer in enumerate(ssd_layers):
    input_shape = np.array(layer.input_shape)
    output_shape = np.array(layer.output_shape)
    kernel_shape = np.array(layer.kernel_shape)

    print()
    print(input_shape)
    print(output_shape)
    # print(kernel_shape)

    box_unit = np.array([1.0 / input_shape[1], 1.0 / input_shape[2]])
    # print(box_unit)

    padding_unit = _calc_paddings(kernel_shape, [1, 1],
                                  input_shape, output_shape)
    # print(padding_unit)

    padding_left = padding_unit[0] * box_unit[1]
    padding_top = padding_unit[1] * box_unit[0]
    padding_right = padding_unit[2] * box_unit[1]
    padding_bottom = padding_unit[3] * box_unit[0]

    if box_unit[0] == 1.0:
      box_size = box_unit
      box_stride = box_unit
    else:
      box_size = box_unit * kernel_shape
      box_stride = box_unit

    print(box_size)

    for y in range(output_shape[1]):
      for x in range(output_shape[2]):
        left_pos = -padding_left + x * box_stride[1]
        top_pos = -padding_top + y * box_stride[0]
        right_pos = left_pos + box_size[1]
        bottom_pos = top_pos + box_size[0]
        box = Box(left_pos, top_pos, right_pos, bottom_pos)
        boxes.append(box)

  return boxes


# https://www.tensorflow.org/versions/r1.0/api_guides/python/nn#Convolution
def _calc_paddings(kernels, strides,
                   input_shape, output_shape):
  pad_along_height = max((output_shape[0] - 1) * strides[0] +
                         kernels[0] - input_shape[0], 0)
  pad_along_width = max((output_shape[1] - 1) * strides[1] +
                        kernels[1] - input_shape[1], 0)
  pad_top = pad_along_height // 2
  pad_bottom = pad_along_height - pad_top
  pad_left = pad_along_width // 2
  pad_right = pad_along_width - pad_left

  return (pad_left, pad_top, pad_right, pad_bottom)
