import os
import sys

import tensorflow as tf

import dataset_loader
import model_provider
from common import calc_hnm_loss, get_tfrecord_path_list
from data_divider import separate_train_test_data

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('tfrecord_dir', '/Users/keiji_ariyama/train2017_tfrecords', "")
tf.flags.DEFINE_string('train_dir', './train', "")
tf.flags.DEFINE_string('summary_dir', './summary', "")

tf.flags.DEFINE_integer('batch_size', 64, "")
tf.flags.DEFINE_float('learning_rate', 0.0001, "")
tf.flags.DEFINE_integer('hnm_ratio', 3,
                        "Hard Negative Mining数の係数 \npositive_sample * hnm_ratio")

tf.flags.DEFINE_integer('step', 100000, "")

SAVE_BY_STEPS = 1000
SHUFFLE_BUFFER_SIZE = 10000

def main(argv=None):
  assert FLAGS.tfrecord_dir, 'tfrecord_dir not set'
  assert os.path.exists(FLAGS.tfrecord_dir), '%s is not exist' % FLAGS.tfrecord_dir

  model = model_provider.get_model()

  train_dir = os.path.join(FLAGS.train_dir, model.NAME)
  os.makedirs(train_dir, exist_ok=True)
  checkpoint_path = os.path.join(train_dir, 'model.ckpt')

  summary_dir = os.path.join(FLAGS.summary_dir, model.NAME)
  summary_dir = os.path.join(summary_dir, 'train')
  os.makedirs(summary_dir, exist_ok=True)

  global_step = tf.Variable(0, trainable=False)

  tfrecord_path_list = get_tfrecord_path_list(FLAGS.tfrecord_dir)
  train_path_list, _ = separate_train_test_data(tfrecord_path_list)

  dataset = dataset_loader \
    .load_dataset(train_path_list, model) \
    .shuffle(SHUFFLE_BUFFER_SIZE) \
    .repeat() \
    .batch(FLAGS.batch_size)
  iterator = dataset.make_one_shot_iterator()

  _, label_masks_batch, boxes_batch, image_batch = iterator.get_next()
  tf.summary.image('images', image_batch, max_outputs=4)

  feature_map = model.base_layers(image_batch)
  ssd_logits = model.ssd_layers(feature_map)

  loss = calc_hnm_loss(boxes_batch, ssd_logits, label_masks_batch,
                       hnm_ratio=FLAGS.hnm_ratio)
  tf.summary.scalar('loss', loss)

  lr = FLAGS.learning_rate
  # lr = tf.train.exponential_decay(lr, global_step, 150000, 0.1, staircase=True)
  tf.summary.scalar('learning_rate', lr)

  opt = tf.train.AdamOptimizer(lr)

  # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
  # http://mickey24.hatenablog.com/entry/2017/07/07/tensorflow-batch-norm-pitfalls
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = opt.minimize(loss, global_step=global_step)

  saver = tf.train.Saver(var_list=tf.global_variables(),
                         max_to_keep=3)
  summary_op = tf.summary.merge_all()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(train_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
      saver.restore(sess, checkpoint.model_checkpoint_path)

    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    current_step = sess.run(global_step)
    max_step = FLAGS.step + current_step

    while (current_step < max_step):
      current_step, loss_value, _ = sess.run([global_step, loss, train_op])
      if current_step % SAVE_BY_STEPS == 0:
        print('Step: %d, loss: %.4f' % (current_step, loss_value))
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, current_step)
        saver.save(sess, checkpoint_path, global_step=current_step)

    open(os.path.join(train_dir, 'complete-%d' % max_step), mode='w').close()


if __name__ == "__main__":
  main(sys.argv)
