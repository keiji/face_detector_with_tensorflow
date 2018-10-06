import os

import tensorflow as tf


def get_tfrecord_path_list(dir):
  result = []
  if not os.path.exists(dir):
    return result

  for file in os.listdir(dir):
    path = os.path.join(dir, file)

    if os.path.isdir(path):
      result.extend(get_tfrecord_path_list(path))
    elif path.endswith('tfrecord'):
      result.append(path)

  return result


NEGATIVE_RATIO = 3


def _calc_hnm_loss(loss, positive_mask, hnm_ratio):
  shape = tf.shape(loss)

  positive_mask = tf.reshape(positive_mask, [shape[0], shape[1]])
  loss = tf.reduce_sum(loss, axis=2)

  positive_count = tf.cast(tf.reduce_sum(positive_mask), tf.int32)
  positive_count = tf.maximum(1, positive_count)

  negative_count = positive_count * hnm_ratio
  negative_count = tf.minimum(shape[1], negative_count)
  negative_count = tf.maximum(1, negative_count)

  positive_losses = loss * positive_mask
  negative_losses = loss - positive_losses

  negative_losses, _ = tf.nn.top_k(negative_losses, k=negative_count)

  positive_losses = tf.reduce_sum(positive_losses)
  negative_losses = tf.reduce_sum(negative_losses)

  positive_loss = positive_losses / tf.cast(positive_count, tf.float32)
  negative_loss = negative_losses / tf.cast(negative_count, tf.float32)

  return positive_loss, negative_loss


def _smooth_l1_loss(x):
  with tf.name_scope("smooth_l1"):
    abs_x = tf.abs(x)
    less_mask = tf.cast(abs_x < 1.0, tf.float32)

    return less_mask * (0.5 * tf.square(x)) + (1.0 - less_mask) * (abs_x - 0.5)


def calc_hnm_loss(ground_truth, logits, positive_mask,
                  hnm_ratio=NEGATIVE_RATIO):
  with tf.name_scope('calc_hnm_loss'):
    logits_offset = logits[:, :, :4]
    logits_classes = logits[:, :, 4:]

    gt_offset = ground_truth[:, :, :4]
    gt_classes = ground_truth[:, :, 4:]

    loss_classes = tf.squared_difference(
      gt_classes,
      tf.nn.sigmoid(logits_classes))
    positive_loss_classes, negative_loss_classes = _calc_hnm_loss(loss_classes,
                                                                  positive_mask,
                                                                  hnm_ratio)
    loss_classes = (positive_loss_classes + negative_loss_classes) / 2.0
    tf.summary.scalar('loss/loss_classes', loss_classes)

    loss_offset = gt_offset - tf.nn.tanh(logits_offset)

    smooth_l1_offset = _smooth_l1_loss(loss_offset)

    smooth_l1_offset_sum = tf.reduce_sum(smooth_l1_offset, axis=2)
    smooth_l1_offset_sum = tf.expand_dims(smooth_l1_offset_sum, axis=2)
    smooth_l1_offset_sum = smooth_l1_offset_sum * positive_mask
    tf.summary.histogram('loss/smooth_l1_offset', smooth_l1_offset_sum)

    smooth_l1_offset = tf.reduce_sum(smooth_l1_offset_sum)
    tf.summary.scalar('loss/loss_offset', smooth_l1_offset)

    return smooth_l1_offset + loss_classes
