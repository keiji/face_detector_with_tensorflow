import json
import math
import os

from absl import app
from absl import flags

import PIL
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


COUNT_PER_PAGE = 1000


def main(argv=None):
  os.makedirs(FLAGS.output_dir, exist_ok=True)

  annotation_file_list = os.listdir(FLAGS.annotation_dir)
  annotation_file_list = sorted(annotation_file_list)

  annotation_file_list = filter(lambda f: f.endswith('.json'), annotation_file_list)
  annotation_file_path_list = list(map(lambda f: os.path.join(FLAGS.annotation_dir, f), annotation_file_list))

  annotation_file_count = len(annotation_file_path_list)

  page_count = math.ceil(annotation_file_count / COUNT_PER_PAGE)

  for page in range(page_count):
    start = COUNT_PER_PAGE * page
    end = min(start + COUNT_PER_PAGE, annotation_file_count)
    annotation_file_path_list_per_page = annotation_file_path_list[start:end]
    output_image_dir = os.path.join(FLAGS.output_dir, 'data%d' % page)
    os.makedirs(output_image_dir, exist_ok=True)

    result_path = os.path.join(FLAGS.output_dir, 'data%d.json' % page)

    frames = {}

    for index, annotation_path in enumerate(annotation_file_path_list_per_page):
      print(
        '%d/%d (%d/%d page): %s' % (index, len(annotation_file_path_list_per_page), page, page_count, annotation_path))

      with open(annotation_path) as fp:
        json_array = json.load(fp)
        if len(json_array) == 0:
          continue

        image_file_name = json_array[0]['file_name']
        image_file_path = os.path.join(FLAGS.image_dir, image_file_name)
        output_image_path = os.path.join(output_image_dir, image_file_name)

        image = Image.open(image_file_path, mode='r').convert('RGB')
        image = _resize(image)
        if not os.path.exists(output_image_path):
          image.save(output_image_path, format='JPEG')

        frame = []

        for id, rect in enumerate(json_array):
          for tag in tags:
            if tag not in rect:
              continue
            result = {}
            result['id'] = id
            result['name'] = id + 1
            result['type'] = 'Rectangle'
            result['width'] = image.width
            result['height'] = image.height
            result['tags'] = [tag]
            result['y1'] = int(rect[tag]['top'] * image.height)
            result['x1'] = int(rect[tag]['left'] * image.width)
            result['y2'] = int(rect[tag]['bottom'] * image.height)
            result['x2'] = int(rect[tag]['right'] * image.width)

            if result['x1'] == result['x2']:
              a = 1 / image.width
              result['x1'] -= a
              result['x1'] = result['x1'] if result['x1'] >= 0 else 0
              result['x2'] += a
              result['x2'] = result['x2'] if result['x2'] <= image.width else image.width

            if result['y1'] == result['y2']:
              a = 1 / image.height
              result['y1'] -= a
              result['y1'] = result['y1'] if result['y1'] >= 0 else 0
              result['y2'] += a
              result['y2'] = result['y2'] if result['y2'] <= image.height else image.height

            frame.append(result)

      frames[str(index)] = frame

    result_object = {}
    result_object['frames'] = frames
    result_object['framerate'] = '1'
    result_object['inputTags'] = ','.join(tags)
    result_object['suggestiontype'] = 'track'
    result_object['scd'] = False
    result_object['visitedFrames'] = []
    result_object['tag_colors'] = ['#0ce1ee']

    with open(result_path, mode='w') as result_fp:
      json.dump(result_object, result_fp, indent=4, sort_keys=True)


# main
if __name__ == "__main__":
  app.run(main)
