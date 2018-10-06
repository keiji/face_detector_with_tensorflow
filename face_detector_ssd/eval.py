import glob
import math
import os
import shutil
import sys
import time

import tensorflow as tf

import dataset_loader
import model_provider
from common import calc_hnm_loss, get_tfrecord_path_list
from data_divider import separate_train_test_data

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('tfrecord_dir', '/Users/keiji_ariyama/train2017_tfrecords', "")
tf.flags.DEFINE_string('train_dir', './train', "")
tf.flags.DEFINE_string('summary_dir', './summary', "")

tf.flags.DEFINE_integer('batch_size', 32, "")
tf.flags.DEFINE_integer('hnm_ratio', 2,
                        "Hard Negative Mining数の係数 \npositive_sample * hnm_ratio")


def _eval(checkpoint,
          saver,
          summary_writer, summary_op,
          global_step, loss,
          num_iter,
          prev_step):
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, checkpoint.model_checkpoint_path)

    current_step = sess.run(global_step)
    if current_step == prev_step:
      return current_step, -1

    loss_values = []

    for iter in range(num_iter):
      loss_value = sess.run(loss)
      loss_values.append(loss_value)

    average_loss_value = sum(loss_values) / len(loss_values)

    summary = tf.Summary()
    summary.value.add(tag='loss', simple_value=average_loss_value)
    summary_writer.add_summary(summary, current_step)

    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, current_step)

    return current_step, average_loss_value


MAX_TO_KEEP = 2


def _clear_useless_checkpoints(checkpoint, max_to_keep=MAX_TO_KEEP):
  checkpoint_dir = os.path.dirname(checkpoint.model_checkpoint_path)

  loss_dirs = filter(lambda d: d.startswith('loss_'), os.listdir(checkpoint_dir))
  loss_dirs = sorted(loss_dirs, key=lambda name: float(name[len('loss_'):]))

  path_list = map(lambda f: os.path.join(checkpoint_dir, f), loss_dirs)
  path_list = list(filter(lambda path: os.path.isdir(path), path_list))

  if len(path_list) <= max_to_keep:
    return

  for path in path_list[max_to_keep:]:
    print('%s will be deleted.' % path)
    shutil.rmtree(path)


def _backup_checkpoint(checkpoint, loss_value):
  checkpoint_dir = os.path.dirname(checkpoint.model_checkpoint_path)

  dest_dir = os.path.join(checkpoint_dir, 'loss_%f' % loss_value)
  os.makedirs(dest_dir, exist_ok=True)

  for file in glob.glob(checkpoint.model_checkpoint_path + '*'):
    shutil.copy(file, dest_dir)


def main(argv=None):
  assert FLAGS.tfrecord_dir, 'tfrecord_dir not set'
  assert os.path.exists(FLAGS.tfrecord_dir), '%s is not exist' % FLAGS.tfrecord_dir

  model = model_provider.get_model()
  batch_size = FLAGS.batch_size

  train_dir = os.path.join(FLAGS.train_dir, model.NAME)
  os.makedirs(train_dir, exist_ok=True)

  summary_dir = os.path.join(FLAGS.summary_dir, model.NAME)
  summary_dir = os.path.join(summary_dir, 'eval')
  os.makedirs(summary_dir, exist_ok=True)

  tfrecord_path_list = get_tfrecord_path_list(FLAGS.tfrecord_dir)
  _, test_path_list = separate_train_test_data(tfrecord_path_list)

  sample_count = len(test_path_list)
  num_iter = math.ceil(sample_count / batch_size)

  with tf.Graph().as_default() as g:
    global_step = tf.Variable(0, trainable=False)

    dataset = dataset_loader \
      .load_dataset(test_path_list, model, is_train=False) \
      .repeat() \
      .batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    _, label_masks_batch, boxes_batch, image_batch = iterator.get_next()

    feature_map = model.base_layers(image_batch, is_train=False)
    ssd_logits = model.ssd_layers(feature_map, is_train=False)

    loss = calc_hnm_loss(boxes_batch, ssd_logits, label_masks_batch,
                         hnm_ratio=FLAGS.hnm_ratio)

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(summary_dir, g)

    saver = tf.train.Saver(var_list=tf.global_variables(),
                           max_to_keep=3)

    best_loss_value = -1
    prev_step = -1

    while True:
      checkpoint = tf.train.get_checkpoint_state(train_dir)
      if checkpoint is None or checkpoint.model_checkpoint_path is None:
        print('checkpoint not found.')
        time.sleep(10)
        continue

      prev_step, loss_value = _eval(checkpoint,
                                    saver,
                                    summary_writer, summary_op,
                                    global_step, loss,
                                    num_iter,
                                    prev_step)

      if loss_value == -1:
        time.sleep(10)
        continue

      if best_loss_value == -1 or best_loss_value > loss_value:
        best_loss_value = loss_value
        print('best_loss_value = %f' % best_loss_value)
        _backup_checkpoint(checkpoint, best_loss_value)
        _clear_useless_checkpoints(checkpoint)

      time.sleep(10)


if __name__ == "__main__":
  main(sys.argv)
