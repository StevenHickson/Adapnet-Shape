import matplotlib
import matplotlib.cm
import numpy as np
import tensorflow as tf
import math

def extract_labels(labels):
    return tf.math.argmax(labels, axis=-1)

def extract_normals(normals):
    return  tf.clip_by_value(tf.nn.l2_normalize(normals, axis=-1), -1.0, 1.0)

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    value = tf.cast(value, tf.float32)
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = cm(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.cast(tf.gather(colors, indices) * 255, tf.uint8)

    return value

def colorize_normals(value):
    value = (value + 1) * 127.5
    return tf.cast(value, tf.uint8)

def add_metric_summaries(images=None,
                        images_estimate=None,
                        depth=None,
                        depth_estimate=None,
                        normals=None,
                        normals_estimate=None,
                        depth_weights=None,
                        labels=None,
                        labels_estimate=None,
                        num_label_classes=None):
    update_ops = []
    if normals_estimate is not None:
        dist = 1 - tf.losses.cosine_distance(normals, normals_estimate, axis=-1, weights=depth_weights, reduction=tf.losses.Reduction.NONE)
        dist_angle = 180.0 / math.pi * tf.math.acos(dist)
        #num_samples = float(int(config['width']) * int(config['height']))
        parsed_angle = tf.boolean_mask(dist_angle, tf.is_finite(dist_angle))
        metric1, update_op1 = tf.metrics.percentage_below(dist_angle, 11.25, weights=depth_weights)
        metric2, update_op2 = tf.metrics.percentage_below(dist_angle, 22.5, weights=depth_weights)
        metric3, update_op3 = tf.metrics.percentage_below(dist_angle, 30, weights=depth_weights)
        tf.summary.scalar('under_11.25', metric1)
        tf.summary.scalar('under_22.5', metric2)
        tf.summary.scalar('under_30', metric3)
        tf.summary.scalar('mean_angle', tf.reduce_mean(parsed_angle))
        update_ops += [update_op1, update_op2, update_op3]
    if labels_estimate is not None:
        mean_iou, mean_update_op = tf.metrics.mean_iou(labels=labels, predictions=labels_estimate, num_classes=num_label_classes)
        tf.summary.scalar('m_iou', mean_iou)
        update_ops += [mean_update_op]
    return update_ops

def add_image_summaries(images=None,
                        images_estimate=None,
                        depth=None,
                        depth_estimate=None,
                        normals=None,
                        normals_estimate=None,
                        labels=None,
                        labels_estimate=None,
                        num_label_classes=None):
    if images is not None:
        tf.summary.image('rgb', images)
    if images_estimate is not None:
        tf.summary.image('rgb_estimate', images_estimate)
    if normals is not None:
        tf.summary.image('normals', colorize_normals(normals))
    if normals_estimate is not None:
        tf.summary.image('normals_estimate', colorize_normals(normals_estimate))
    if depth is not None:
        tf.summary.image('depth', colorize(depth, cmap='jet'))
    if depth_estimate is not None:
        tf.summary.image('depth_estimate', colorize(depth_estimate, cmap='jet'))
    if labels is not None:
        tf.summary.image('label', colorize(labels, cmap='jet', vmin=0, vmax=num_label_classes))
    if labels_estimate is not None:
        tf.summary.image('labels_estimate',  colorize(labels_estimate, cmap='jet', vmin=0, vmax=num_label_classes))
