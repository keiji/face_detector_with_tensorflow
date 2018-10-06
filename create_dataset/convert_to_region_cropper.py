import datetime
import json
import os

import PIL

from absl import app
from absl import flags

from PIL import Image

FLAGS = flags.FLAGS
flags.DEFINE_string('image_dir', None, "処理する画像があるディレクトリ")
flags.DEFINE_string('annotation_dir', None, "処理するアノテーションがあるディレクトリ")
flags.DEFINE_string('output_dir', None, "出力先ディレクトリ")

tags = [
  'position',
  'left_eye', 'right_eye',
  'left_cheek', 'right_cheek',
  'nose_base',
  'left_mouth', 'right_mouth', 'bottom_mouth',
]


def _resize(image):
  size = max(image.width, image.height)
  ratio = 256 / size
  image_size = (round(image.width * ratio), round(image.height * ratio))
  return image.resize(image_size, PIL.Image.LANCZOS)


def main(argv=None):
  assert FLAGS.image_dir, '--image_dir is not set'
  assert os.path.exists(FLAGS.image_dir), '--image_dir %s is not exist' % FLAGS.image_dir

  assert FLAGS.annotation_dir, '--annotation_dir is not set'
  assert os.path.exists(FLAGS.annotation_dir), '--annotation_dir %s is not exist' % FLAGS.annotation_dir

  assert FLAGS.output_dir, '--output_dir is not set'
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  annotation_file_list = os.listdir(FLAGS.annotation_dir)
  annotation_file_list = sorted(annotation_file_list)

  annotation_file_list = filter(lambda f: f.endswith('.json'), annotation_file_list)
  annotation_file_path_list = list(
    map(lambda f: os.path.join(FLAGS.annotation_dir, f), annotation_file_list))

  annotation_file_count = len(annotation_file_path_list)

  for index, annotation_path in enumerate(annotation_file_path_list):
    print(
      '%d/%d: %s' % (index + 1, annotation_file_count, annotation_path))

    annotation_file_name = os.path.basename(annotation_path)
    output_file_path = os.path.join(FLAGS.output_dir, annotation_file_name)

    with open(annotation_path) as fp:
      json_array = json.load(fp)
      if len(json_array) == 0:
        continue

      image_file_name = json_array[0]['file_name']
      image_file_path = os.path.join(FLAGS.image_dir, image_file_name)

      image = Image.open(image_file_path, mode='r').convert('RGB')
      image = _resize(image)

      output_image_path = os.path.join(FLAGS.output_dir, image_file_name)
      if not os.path.exists(output_image_path):
        image.save(output_image_path, format='JPEG')

      regions = []

      for id, rect in enumerate(json_array):
        for tag in tags:
          if tag not in rect:
            continue
          region = {}
          region['probability'] = 0.999
          region['label'] = tags.index(tag)

          top = rect[tag]['top']
          left = rect[tag]['left']
          bottom = rect[tag]['bottom']
          right = rect[tag]['right']

          top = top if top >= 0.0 else 0.0
          left = left if left >= 0.0 else 0.0
          bottom = bottom if bottom <= 1.0 else 1.0
          right = right if right <= 1.0 else 1.0

          region['rect'] = {
            'left': left,
            'right': right,
            'top': top,
            'bottom': bottom,
          }
          regions.append(region)

      result_object = {}
      result_object['regions'] = regions
      result_object['file_name'] = image_file_name
      result_object['generator'] = 'convert_to_region_cropper.py'
      result_object['labels'] = tags

      # 2018-08-27T06:38:45.420
      result_object['created_at'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

      with open(output_file_path, mode='w') as result_fp:
        json.dump(result_object, result_fp, indent=4, sort_keys=True)


# main
if __name__ == "__main__":
  app.run(main)
