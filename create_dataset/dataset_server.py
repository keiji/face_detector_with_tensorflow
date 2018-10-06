import io
import json
import os
from http import HTTPStatus

import PIL
from PIL import Image
from flask import Flask, request, Response

app = Flask(__name__)

from absl import app as app_absl
from absl import flags

from PIL import Image

FLAGS = flags.FLAGS
flags.DEFINE_string('image_dir', None, "送信する画像があるディレクトリ")
flags.DEFINE_string('output_dir', None, "受信したJSONを保存するディレクトリ")


@app.route('/')
def index():
  offset = int(request.args.get('offset', 0))
  limit = int(request.args.get('limit', -1))

  if offset > 0 and limit == -1:
    limit = 100

  files = os.listdir(FLAGS.image_dir)

  if offset > 0 or limit > 0:
    start = offset
    start = 0 if start < 0 else start

    end = offset + limit
    end = len(files) if end > len(files) else end

    print('%d, %d' % (start, end))

    if start >= len(files) or end > len(files):
      return [], HTTPStatus.OK

    files = files[start:end]

  return json.dumps(files)


@app.route('/<name>', methods=['PUT'])
def put_annotation(name):
  path = os.path.join(FLAGS.image_dir, name)
  if not os.path.exists(path):
    return '{"error" : "not found"}', HTTPStatus.NOT_FOUND

  print(request.json)

  name, ext = os.path.splitext(name)
  json_file = '%s.json' % name

  json_path = os.path.join(FLAGS.output_dir, json_file)
  with open(json_path, mode='w') as fp:
    json.dump(request.json, fp, indent=4)

  return '{}', HTTPStatus.OK


@app.route('/<name>')
def file(name):
  path = os.path.join(FLAGS.image_dir, name)
  if not os.path.exists(path):
    return '[]', HTTPStatus.NOT_FOUND

  return _get_image(path, 256)


def _get_image(file_path, request_size):
  image = Image.open(file_path, mode='r').convert('RGB')

  if request_size > 0:
    size = max(image.width, image.height)
    ratio = request_size / size
    image_size = (round(image.width * ratio), round(image.height * ratio))
    image = image.resize(image_size, PIL.Image.LANCZOS)

  image_bytes = io.BytesIO()
  image.save(image_bytes, format='JPEG')

  return Response(image_bytes.getvalue(), mimetype='image/jpeg')


def main(argv):
  assert FLAGS.image_dir, '--image_dir is not set.'
  assert os.path.exists(FLAGS.image_dir), '--image_dir %s is not exist.' % FLAGS.image_dir

  assert FLAGS.output_dir, '--output_dir is not set.'
  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

  host = os.getenv("APP_ADDRESS", '0.0.0.0')
  port = os.getenv("APP_PORT", 3000)

  app.run(host, port)


# main
if __name__ == "__main__":
  app_absl.run(main)
