import os

import hashlib


def separate_train_test_data(tfrecord_path_list):
  train_path_list = []
  test_path_list = []

  for tfrecord_path in tfrecord_path_list:
    file_name = os.path.basename(tfrecord_path)
    name, ext = os.path.splitext(file_name)
    base = name.split('_')[0]

    hash = hashlib.sha1(base.encode('utf-8')).digest()
    index = sum(map(lambda n: int(n), hash)) % 10
    if index > 0:
      train_path_list.append(tfrecord_path)
    else:
      test_path_list.append(tfrecord_path)

  print('train data: %d' % (len(train_path_list)))
  print('test data: %d' % (len(test_path_list)))

  return tfrecord_path_list, test_path_list
