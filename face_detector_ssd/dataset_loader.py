import tensorflow as tf


def load_dataset(tfrecord_path_list, model, is_train=True):
  def _read_tfrecord(example_proto):
    features = tf.parse_single_example(
      example_proto,
      features={
        'tfrecord_name': tf.FixedLenFeature([], tf.string),
        'image_file_name': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
        'labels': tf.VarLenFeature(tf.float32),
        'label_masks': tf.VarLenFeature(tf.float32),
        'offset': tf.VarLenFeature(tf.float32),
      })

    tfrecord_name = features['tfrecord_name']
    raw_image = features['image']

    label_masks = features['label_masks'].values
    label_masks = tf.reshape(label_masks, [-1, 1])

    offset = features['offset'].values
    offset = tf.reshape(offset, [-1, 4])

    labels = features['labels'].values
    labels = tf.reshape(labels, [-1, 1])

    boxes = tf.concat((offset, labels), axis=1)

    return tfrecord_name, label_masks, boxes, raw_image

  def _process_record(tfrecord_name, label_masks, boxes, raw_image):
    image = tf.image.decode_jpeg(raw_image, channels=3)
    image = tf.cast(image, tf.float32)

    image = tf.image.resize_images(image, (model.IMAGE_SIZE, model.IMAGE_SIZE))

    return tfrecord_name, label_masks, boxes, image

  def _distort(tfrecord_name, label_masks, boxes, image):
    distorted_image = tf.image.random_brightness(image, max_delta=60)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               upper=1.5, lower=0.2)
    distorted_image = tf.image.random_saturation(distorted_image,
                                                 upper=1.5, lower=0.2)

    return tfrecord_name, label_masks, boxes, distorted_image

  def _normalize(tfrecord_name, label_masks, boxes, image):
    image = image / 255.0
    return tfrecord_name, label_masks, boxes, image

  dataset = tf.data.TFRecordDataset(tfrecord_path_list,
                                    compression_type='GZIP') \
    .map(_read_tfrecord) \
    .map(_process_record)

  if is_train:
    dataset = dataset.map(_distort)

  dataset = dataset.map(_normalize)

  return dataset
