import json
import os
import sys
from io import BytesIO

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

from jaccard_overlap import jaccard_overlap

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('base_dir', None,
                       "処理対象の画像とJSONがある起点ディレクトリ")
tf.flags.DEFINE_string('output_dir', None,
                       "出力先ディレクトリ")
tf.flags.DEFINE_boolean('debug', False,
                        "デバッグ用に矩形を表示した画像を出力する")

from playground.plan_boundingbox import create_ssd_layers, create_boxes

# from playground.plan_boundingbox2 import create_ssd_layers, create_boxes
# from playground.plan_boundingbox_lightweight import create_ssd_layers, create_boxes

IMAGE_SIZE = 256
FEATURE_MAP_SHAPE = [1, 16, 16, 3]
THRESHOLD_OVERLAP = 0.4


def _calc_jaccard_overlap(face, box):
  box_rect = (box.left, box.top, box.right, box.bottom)
  face_rect = (face['left'], face['top'], face['right'], face['bottom'])
  overlap = jaccard_overlap(face_rect, box_rect)
  return overlap


def _assign_box(face_regions, boxes):
  face_count = len(face_regions)
  match_count = 0

  for face_region in face_regions:
    rect = face_region['rect']

    sorted_boxes = sorted(boxes,
                          key=lambda box: _calc_jaccard_overlap(rect, box),
                          reverse=True)

    matched = False
    for index, box in enumerate(sorted_boxes):
      if box.label_mask == 1:
        continue

      overlap = _calc_jaccard_overlap(rect, box)
      if overlap == 0:
        continue

      if (index == 0 or overlap >= THRESHOLD_OVERLAP):
        if not matched:
          match_count += 1
          matched = True

        box.label_mask = 1
        box.label = [1.0]
        box.offset = [
          rect['left'] - box.left,
          rect['top'] - box.top,
          rect['right'] - box.right,
          rect['bottom'] - box.bottom,
        ]

  return face_count, match_count


def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _create_tfrecord(image, image_file_name, boxes, tfrecord_path):
  byte_io = BytesIO()
  image.save(byte_io, format="jpeg")
  byte_io.seek(0)
  jpeg_image_data = byte_io.read()

  tfrecord_name = os.path.basename(tfrecord_path)

  label_masks = np.array(list(
    map(lambda box: box.label_mask, boxes))).flatten()
  labels = np.array(list(map(lambda box: box.label, boxes))).flatten()
  offsets = np.array(list(map(lambda box: box.offset, boxes))).flatten()

  options = tf.python_io.TFRecordOptions(
    tf.python_io.TFRecordCompressionType.GZIP)

  with tf.python_io.TFRecordWriter(tfrecord_path, options) as writer:
    example = tf.train.Example(features=tf.train.Features(feature={
      'tfrecord_name': _bytes_feature(tfrecord_name.encode()),
      'image_file_name': _bytes_feature(image_file_name.encode()),
      'image': _bytes_feature(jpeg_image_data),
      'labels': _float64_feature(labels),
      'label_masks': _float64_feature(label_masks),
      'offset': _float64_feature(offsets),
    }))
    writer.write(example.SerializeToString())


def _create_image_data(image_path, image_size=IMAGE_SIZE, rotate=None):
  with Image.open(image_path) as image:
    image = image.resize((image_size, image_size)) \
      .convert('RGB')

    if rotate is not None:
      # 反時計回り（counter clockwise）
      image = image.rotate(360 - 90 * rotate)

    return image


def _rotate_clockwise(regions, rotate):
  for region in regions:
    rect = region['rect']

    for i in range(rotate):
      left = rect['left']
      top = rect['top']
      right = rect['right']
      bottom = rect['bottom']
      rect['left'] = 1.0 - bottom
      rect['top'] = left
      rect['right'] = 1.0 - top
      rect['bottom'] = right


def _load_annotation(json_path, rotate=None):
  with open(json_path) as fp:
    annotation = json.load(fp)

    if rotate is not None:
      _rotate_clockwise(annotation['regions'], rotate=rotate)

    return annotation


def _create_debug_image(image, boxes, output_image_path):
  draw = ImageDraw.Draw(image)

  for box in boxes:
    if box.label == [0]:
      continue

    rect = [
      (box.left + box.offset[0]) * image.width,
      (box.top + box.offset[1]) * image.height,
      (box.right + box.offset[2]) * image.width,
      (box.bottom + box.offset[3]) * image.height
    ]
    draw.rectangle(rect, outline=0xff0000)

  image.save(output_image_path, format='jpeg')


def main(argv=None):
  assert FLAGS.base_dir, 'Directory base_dir not set.'
  assert os.path.exists(FLAGS.base_dir), \
    'Directory %s not exist.' % FLAGS.base_dir
  os.makedirs(FLAGS.output_dir, exist_ok=True)

  files = os.listdir(FLAGS.base_dir)
  json_files = filter(lambda f: f.endswith('.json'), files)
  json_file_paths = list(
    map(lambda f: os.path.join(FLAGS.base_dir, f), json_files))

  feature_map = tf.get_variable(shape=FEATURE_MAP_SHAPE,
                                name='feature_map')
  ssd_layers = create_ssd_layers(feature_map)

  file_count = len(json_file_paths)

  total_face_count = 0
  total_match_count = 0

  for index, json_path in enumerate(json_file_paths):
    for rotate in range(4):
      annotation = _load_annotation(json_path, rotate=rotate)
      image_file_name = annotation['file_name']

      regions = annotation['regions']
      face_regions = list(
        filter(lambda region: region['label'] == 0, regions))

      boxes = create_boxes(ssd_layers)

      face_count, match_count = _assign_box(face_regions, boxes)
      total_face_count += face_count
      total_match_count += match_count

      image_path = os.path.join(FLAGS.base_dir, image_file_name)
      image = _create_image_data(image_path, rotate=rotate)

      name, _ = os.path.splitext(image_file_name)
      tfrecord_name = '%s_%d.tfrecord' % (name, rotate)
      tfrecord_path = os.path.join(FLAGS.output_dir, tfrecord_name)

      _create_tfrecord(image, image_file_name, boxes, tfrecord_path)

      if FLAGS.debug:
        debug_image_file_name = '%s_%d.jpg' % (name, rotate)
        debug_image_file_path = os.path.join(FLAGS.output_dir,
                                             debug_image_file_name)
        _create_debug_image(image, boxes, debug_image_file_path)

    percentage = total_match_count / float(total_face_count)
    print('%d/%d, %d/%d - %f' % (index + 1, file_count,
                                 total_match_count, total_face_count,
                                 percentage))


if __name__ == "__main__":
  main(sys.argv)
