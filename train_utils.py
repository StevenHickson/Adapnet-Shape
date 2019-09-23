import matplotlib
import matplotlib.cm
import numpy as np
import tensorflow as tf
import math
from dataset.nyu13_dataset import NYU13Dataset
from dataset.nyu20_dataset import NYU20Dataset
from dataset.nyu40_dataset import NYU40Dataset
from dataset.scenenet_dataset import ScenenetDataset
from dataset.scannet_dataset import ScannetDataset

def extract_labels(labels):
    return tf.math.argmax(labels, axis=-1)

def extract_normals(normals):
    return  tf.clip_by_value(tf.nn.l2_normalize(normals, axis=-1), -1.0, 1.0)

def extract_modalities(config):
    modality_infos = []
    num_label_classes = 0
    if 'loss_weights' in config:
        weight_mul_list = config['loss_weights']
    else:
        weight_mul_list = [1.0] * len(config['output_modality'])
    for modality, num_classes, weight_mul in zip(config['output_modality'], config['num_classes'], weight_mul_list):
        modality_infos.append((modality, num_classes, weight_mul))
        if modality == 'labels':
            num_label_classes = num_classes
    return modality_infos, num_label_classes

def calculate_weights(depths, normals):
    valid_depths = tf.math.not_equal(tf.cast(depths, tf.float32), 0)
    valid_normals = tf.expand_dims(tf.math.not_equal(tf.reduce_sum(tf.math.abs(normals), axis=-1), 0), axis=-1)
    return tf.cast(tf.math.logical_and(valid_depths, valid_normals), tf.float32)

def get_dataset(config):
    dataset_name = config['dataset_name']
    if dataset_name == 'nyu13':
        helper = NYU13Dataset()
    elif dataset_name == 'nyu20':
        helper = NYU20Dataset()
    elif dataset_name == 'nyu40':
        helper = NYU40Dataset()
    elif dataset_name == 'scenenet':
        helper = ScenenetDataset()
    elif dataset_name == 'scannet':
        helper = ScannetDataset()
    else:
        print('Non-existant Dataset')
    helper.Setup(config)
    return helper

def setup_model(model, config, train=True):
    images=None
    images_estimate=None
    depth=None
    images_pl=None
    depths_pl=None
    normals_pl=None
    labels_pl=None
    depth_estimate=None
    normals=None
    normals_estimate=None
    labels=None
    labels_estimate=None
    weights = None
    num_label_classes = None

    if config['input_modality'] == 'rgb':
        images_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'], 3])
        images=images_pl
        model_input = images_pl
    elif config['input_modality'] == 'normals':
        normals_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'], 3])
        normals = extract_normals(normals_pl)
        model_input = normals_pl
    elif config['input_modality'] == 'depth':
        depths_pl = tf.placeholder(tf.uint16, [None, config['height'], config['width'], 1])
        depth = tf.cast(depths_pl, tf.float32)
        model_input = tf.tile(depth, [1,1,1,3])
    elif config['input_modality'] == 'depth_notile':
        depths_pl = tf.placeholder(tf.uint16, [None, config['height'], config['width'], 1])
        depth = tf.cast(depths_pl, tf.float32)
        model_input = depth
    
    for modality, num_classes in zip(config['output_modality'], config['num_classes']):
        if modality == 'labels':
            labels_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'],
                                                num_classes])
            labels = extract_labels(labels_pl)
            num_label_classes = num_classes
        elif modality == 'normals':
            normals_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'], 3])
            depths_pl = tf.placeholder(tf.uint16, [None, config['height'], config['width'], 1])
            depth = depths_pl
            normals = extract_normals(normals_pl)
            weights = calculate_weights(depth, normals)
    
    model.build_graph(model_input, depth=depths_pl, label=labels_pl, normals=normals_pl, valid_depths=weights)
    if train:
        model.create_optimizer()
        
        for modality in config['output_modality']:
            if modality == 'labels':
                labels_estimate = extract_labels(model.output_labels)
            elif modality == 'normals':
                normals_estimate = extract_normals(model.output_normals)
      
        add_image_summaries(images=images,
                            images_estimate=images_estimate,
                            depth=depth,
                            depth_estimate=depth_estimate,
                            normals=normals,
                            normals_estimate=normals_estimate,
                            labels=labels,
                            labels_estimate=labels_estimate,
                            num_label_classes=num_label_classes)
        update_ops = add_metric_summaries(images=images,
                                          images_estimate=images_estimate,
                                          depth=depth,
                                          depth_estimate=depth_estimate,
                                          normals=normals,
                                          normals_estimate=normals_estimate,
                                          depth_weights=weights,
                                          labels=labels,
                                          labels_estimate=labels_estimate,
                                          num_label_classes=num_label_classes)

        model._create_summaries()
    else:
        update_ops=tf.no_op()
    return images_pl, depths_pl, normals_pl, labels_pl, update_ops

def setup_model_new(model, data_list, config, train=True):
    images=None
    images_estimate=None
    depth=None
    images_pl=None
    depths_pl=None
    normals_pl=None
    labels_pl=None
    depth_estimate=None
    normals=None
    normals_estimate=None
    labels=None
    labels_estimate=None
    weights = None
    num_label_classes = None

    if config['input_modality'] == 'rgb':
        images_pl = data_list[0]
        images=images_pl
        model_input = images_pl
    elif config['input_modality'] == 'normals':
        normals_pl = data_list[2]
        normals = extract_normals(normals_pl)
        model_input = normals_pl
    elif config['input_modality'] == 'depth':
        depths_pl = data_list[1]
        depths_pl = tf.placeholder(tf.uint16, [None, config['height'], config['width'], 1])
        depth = tf.cast(depths_pl, tf.float32)
        model_input = tf.tile(depth, [1,1,1,3])
    elif config['input_modality'] == 'depth_notile':
        depths_pl = data_list[1]
        depth = tf.cast(depths_pl, tf.float32)
        model_input = depth
    
    for modality, num_classes in zip(config['output_modality'], config['num_classes']):
        if modality == 'labels':
            labels_pl = data_list[3]
            labels = extract_labels(labels_pl)
            num_label_classes = num_classes
        elif modality == 'normals':
            normals_pl = data_list[2]
            depths_pl = data_list[1]
            depth = depths_pl
            normals = extract_normals(normals_pl)
            weights = calculate_weights(depth, normals)
    
    model.build_graph(model_input, depth=depths_pl, label=labels_pl, normals=normals_pl, valid_depths=weights)
    if train:
        
        for modality in config['output_modality']:
            if modality == 'labels':
                labels_estimate = extract_labels(model.output_labels)
            elif modality == 'normals':
                normals_estimate = extract_normals(model.output_normals)
      
        add_image_summaries(images=images,
                            images_estimate=images_estimate,
                            depth=depth,
                            depth_estimate=depth_estimate,
                            normals=normals,
                            normals_estimate=normals_estimate,
                            labels=labels,
                            labels_estimate=labels_estimate,
                            num_label_classes=num_label_classes)
        update_ops = add_metric_summaries(images=images,
                                          images_estimate=images_estimate,
                                          depth=depth,
                                          depth_estimate=depth_estimate,
                                          normals=normals,
                                          normals_estimate=normals_estimate,
                                          depth_weights=weights,
                                          labels=labels,
                                          labels_estimate=labels_estimate,
                                          num_label_classes=num_label_classes)

        model._create_summaries()
    else:
        update_ops=tf.no_op()
    return images_pl, depths_pl, normals_pl, labels_pl, update_ops

def setup_feeddict(data_list, sess, images_pl, depths_pl, normals_pl, labels_pl, config):
    input_names_to_feeds = dict()
    if config['input_modality'] == 'rgb':
        input_names_to_feeds['rgb'] = data_list[0]
    elif config['input_modality'] == 'depth' or config['input_modality'] == 'depth_notile':
        input_names_to_feeds['depth'] = data_list[1]
    elif config['input_modality'] == 'normals':
        input_names_to_feeds['normals'] = data_list[2]

    for modality in config['output_modality']:
        if modality == 'labels':
            input_names_to_feeds['labels'] = data_list[3]
        elif modality == 'depth':
            input_names_to_feeds['depth'] = data_list[1]
        elif modality == 'normals':
            input_names_to_feeds['depth'] = data_list[1]
            input_names_to_feeds['normals'] = data_list[2]

    output_feeds = sess.run(list(input_names_to_feeds.values()))
    feed_dict = dict()
    for feed, name in zip(output_feeds, list(input_names_to_feeds.keys())):
        if name == 'rgb':
            feed_dict[images_pl] = feed
        elif name == 'depth':
            feed_dict[depths_pl] = feed
        elif name == 'normals':
            feed_dict[normals_pl] = feed
        elif name == 'labels':
            feed_dict[labels_pl] = feed
    return feed_dict

def setup_sess_inputs(data_list, inputs, config):
    new_inputs = []
    rgb_added = False
    depth_added = False
    normals_added = False
    labels_added = False

    if config['input_modality'] == 'rgb':
        new_inputs.append(data_list[0])
        rgb_added = True
    elif config['input_modality'] == 'depth' or config['input_modality'] == 'depth_notile':
        new_inputs.append(data_list[1])
        depth_added = True
    elif config['input_modality'] == 'normals':
        new_inputs.append(data_list[2])
        normals_added = True

    for modality in config['output_modality']:
        if modality == 'labels' and not labels_added:
            new_inputs.append(data_list[3])
            labels_added = True
        elif modality == 'depth' and not depth_added:
            new_inputs.append(data_list[1])
            depth_added = True
        elif modality == 'normals'and not normals_added:
            if not depth_added:
                new_inputs.append(data_list[1])
                depth_added = True
            new_inputs.append(data_list[2])
            normals_added = True
    return inputs + new_inputs

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

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
