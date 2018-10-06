import os
import sys
import time

import numpy as np

import tensorflow as tf
from PIL import Image, ImageDraw

from jaccard_overlap import jaccard_overlap
from playground.plan_boundingbox import create_ssd_layers, create_boxes

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('train_path', None, "")
tf.flags.DEFINE_string('image_path', None, "")
tf.flags.DEFINE_string('output_dir', './output', "")

tf.flags.DEFINE_float('confidence_threshold', 0.8, "")

import model.model1 as model

MIN_IMAGE_SIZE = 1024


def _get_image_data(image_path, image_size):
  raw_image = Image.open(image_path)

  if model.CHANNELS == 1:
    raw_image = raw_image.convert('L')
  else:
    raw_image = raw_image.convert('RGB')

  resized_image = raw_image.resize((image_size, image_size), Image.LANCZOS)

  image_size = raw_image.size
  if image_size[0] > MIN_IMAGE_SIZE or image_size[1] > MIN_IMAGE_SIZE:
    ratio = max(MIN_IMAGE_SIZE / image_size[0], MIN_IMAGE_SIZE / image_size[1])
    size = (int(image_size[0] * ratio), int(image_size[1] * ratio))
    raw_image = raw_image.resize(size, Image.LANCZOS)

  return raw_image, resized_image


def _draw_recognized_faces(image, faces):
  draw = ImageDraw.Draw(image)

  # 認識した領域の描画
  for face in faces:
    if face.confidence < 0:
      continue

    rect = [
      (face.left) * image.width,
      (face.top) * image.height,
      (face.right) * image.width,
      (face.bottom) * image.height
    ]
    draw.rectangle(rect, outline=0x0000ff)

  return image


def _save_image(image, output_path):
  image.save(output_path, format='jpeg')


def _merge_faces(faces, overlap_threshold=0.5):
  faces = sorted(faces,
                 key=lambda face: face.confidence,
                 reverse=True)

  for base_index, base_face in enumerate(faces):
    if base_face.confidence < FLAGS.confidence_threshold:
      continue

    base_rect = [base_face.left,
                 base_face.top,
                 base_face.right,
                 base_face.bottom]

    while True:
      merged_flag = False

      for target_index, target_face in enumerate(faces):
        if base_index == target_index:
          continue
        if target_face.confidence < FLAGS.confidence_threshold:
          continue

        rect = [target_face.left,
                target_face.top,
                target_face.right,
                target_face.bottom]

        overlap = jaccard_overlap(base_rect, rect)

        if overlap >= overlap_threshold:
          total_confidence = base_face.confidence + target_face.confidence
          base_weight = base_face.confidence / total_confidence
          target_weight = 1.0 - base_weight

          base_face.left = (base_face.left * base_weight) \
                           + (target_face.left * target_weight)
          base_face.top = (base_face.top * base_weight) \
                          + (target_face.top * target_weight)
          base_face.right = (base_face.right * base_weight) \
                            + (target_face.right * target_weight)
          base_face.bottom = (base_face.bottom * base_weight) \
                             + (target_face.bottom * target_weight)

          target_face.confidence = -1

          merged_flag = True

      if not merged_flag:
        break


class Face(object):
  def __init__(self, box):
    self.confidence = box.label[0]
    self.left = box.left + box.offset[0]
    self.top = box.top + box.offset[1]
    self.right = box.right + box.offset[2]
    self.bottom = box.bottom + box.offset[3]


def main(argv=None):
  assert FLAGS.train_path, 'train_path not set'

  assert FLAGS.image_path, 'image_path not set'
  assert os.path.exists(FLAGS.image_path), '%s is not exist' % FLAGS.image_path
  print(FLAGS.image_path)

  os.makedirs(FLAGS.output_dir, exist_ok=True)

  raw_image, resized_image = _get_image_data(FLAGS.image_path, model.IMAGE_SIZE)

  image = np.array(resized_image.getdata()) \
    .reshape(model.IMAGE_SIZE, model.IMAGE_SIZE, model.CHANNELS) \
    .astype(np.float32)

  file_name = os.path.basename(FLAGS.image_path)
  name, _ = os.path.splitext(file_name)

  output_image_file_name = '%s.jpg' % name
  output_image_path = os.path.join(FLAGS.output_dir, output_image_file_name)

  output_result_image_file_name = '%s_result.jpg' % name
  output_result_image_path = os.path.join(FLAGS.output_dir,
                                          output_result_image_file_name)

  global_step = tf.Variable(0, trainable=False)

  image_ph = tf.placeholder(tf.float32,
                            [model.IMAGE_SIZE, model.IMAGE_SIZE, model.CHANNELS])
  normalized_image = image_ph * (1 / 255.0)

  image_batch = tf.expand_dims(normalized_image, axis=0)

  feature_map = model.base_layers(image_batch, is_train=False)
  ssd_logits = model.ssd_layers(feature_map, is_train=False)

  loc_logits = ssd_logits[:, :, :4]
  conf_logits = ssd_logits[:, :, 4:]

  loc_logits = tf.nn.tanh(loc_logits)
  conf_logits = tf.nn.sigmoid(conf_logits)

  ssd_logits = tf.concat([loc_logits, conf_logits], axis=2)

  saver = tf.train.Saver(var_list=tf.global_variables(),
                         max_to_keep=3)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, FLAGS.train_path)

    start = time.time()

    ssd_logits_values = sess.run(ssd_logits,
                                 feed_dict={
                                   image_ph: image
                                 })

    elapsed = time.time() - start
    print('Inference Elapsed %f s' % elapsed)

    start = time.time()

    ssd_layers = create_ssd_layers(feature_map)
    boxes = create_boxes(ssd_layers)

    for index, box_value in enumerate(ssd_logits_values[0]):
      boxes[index].label = box_value[4:]
      boxes[index].offset = box_value[:4]

    filtered_boxes = filter(
      lambda box: box.label[0] > FLAGS.confidence_threshold,
      boxes)

    faces = list(map(lambda box: Face(box), filtered_boxes))
    _merge_faces(faces, overlap_threshold=0.3)

    elapsed = time.time() - start
    print('Post-process Elapsed %f s' % elapsed)

    result_image = _draw_recognized_faces(raw_image.copy(), faces)

    result_image.save(output_result_image_path, format='jpeg')
    raw_image.save(output_image_path, format='jpeg')


if __name__ == "__main__":
  main(sys.argv)
