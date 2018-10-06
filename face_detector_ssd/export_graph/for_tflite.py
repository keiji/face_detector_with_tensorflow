# coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow as tf

INPUT_CHANNEL = 4

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('train_path', None, "学習結果のパス")
tf.flags.DEFINE_string('output_dir', None, "出力先のディレクトリ")

from playground.plan_boundingbox import create_ssd_layers, create_boxes
from model import model1 as model


def _export_graph(sess):
  constant_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, sess.graph_def, ['offset', 'confidence'])

  tf.train.write_graph(constant_graph_def, FLAGS.output_dir,
                       '%s_4ch.pb' % model.NAME, as_text=False)

def _export_boxes_position(feature_map, output_dir):
  box_layers = create_ssd_layers(feature_map)
  boxes = create_boxes(box_layers)

  box_position = []
  for index, box in enumerate(boxes):
    box_position.append({
      'index': index,
      'left': box.left,
      'top': box.top,
      'right': box.right,
      'bottom': box.bottom,
    })

  file_name = '%s_boxes_position.json' % model.NAME
  box_position_path = os.path.join(output_dir, file_name)
  with open(box_position_path, mode='w') as fp:
    json.dump(box_position, fp, indent=4)

def main(args=None):
  assert FLAGS.train_path, 'train_path is not set.'
  assert FLAGS.output_dir, 'output_dir is not set.'

  with tf.Graph().as_default() as g:
    image_ph = tf.placeholder(
      tf.float32,
      [model.IMAGE_SIZE * model.IMAGE_SIZE * INPUT_CHANNEL],
      name='input')

    image = tf.reshape(
      image_ph,
      [model.IMAGE_SIZE, model.IMAGE_SIZE, INPUT_CHANNEL])


    # アルファチャンネルを削除
    image = image[:, :, :3]

    normalized_image = tf.multiply(image, 1 / 255.0)
    normalized_image = tf.expand_dims(normalized_image, axis=0)

    feature_map = model.base_layers(normalized_image, is_train=False)
    ssd_logits = model.ssd_layers(feature_map, is_train=False)
    ssd_logits = tf.reshape(ssd_logits, [-1, model.OFFSET + model.CLASSES])

    location_offset = tf.nn.tanh(ssd_logits[:, :4], name='offset')
    confidence = tf.nn.sigmoid(ssd_logits[:, 4:], name='confidence')

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
      saver.restore(sess, FLAGS.train_path)

      _export_graph(sess)
      _export_boxes_position(feature_map, FLAGS.output_dir)



if __name__ == '__main__':
  tf.app.run()
