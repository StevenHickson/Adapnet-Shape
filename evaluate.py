''' AdapNet++:  Self-Supervised Model Adaptation for Multimodal Semantic Segmentation

 Copyright (C) 2018  Abhinav Valada, Rohit Mohan and Wolfram Burgard

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.'''

import argparse
import datetime
import importlib
import os
import numpy as np
import tensorflow as tf
import yaml
import math
from train_utils import *

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-c', '--config', default='config/cityscapes_test.config')

def compute_label_matrix(label_max, pred_max, output_matrix):
    # Input:
    # label_max shape(B,H,W): np.argmax(one_hot_encoded_label,3)
    # pred_max shape(B,H,W): np.argmax(softmax,3)
    # output_matrix shape(NUM_CLASSES,3): if func is called first time an array of 
    #                                     zeros.
    # Output:
    # output_matrix shape(NUM_CLASSES,3): columns with total count of true positives,
    #                                     false positives and false negatives.
    for i in xrange(output_matrix.shape[0]):
        temp = pred_max == i
        temp_l = label_max == i
        tp = np.logical_and(temp, temp_l)
        temp[temp_l] = True
        fp = np.logical_xor(temp, temp_l)
        temp = pred_max == i
        temp[fp] = False
        fn = np.logical_xor(temp, temp_l)
        output_matrix[i, 0] += np.sum(tp)
        output_matrix[i, 1] += np.sum(fp)
        output_matrix[i, 2] += np.sum(fn)

    return output_matrix

def compute_iou(output_matrix):
    # Input:
    # output_matrix shape(NUM_CLASSES,3): columns with total count of true positives,
    #                                     false positives and false negatives.
    # Output:
    # IoU in percent form (doesn't count label id 0 contribution as it is assumed to be void) 
    return np.sum(output_matrix[1:, 0]/(np.sum(output_matrix[1:, :], 1).astype(np.float32)+1e-10))/(output_matrix.shape[0]-1)*100

def get_label_metrics(probabilities, feed_dict, labels_pl, labels_matrix):
    prediction = np.argmax(probabilities, axis=-1)
    label = feed_dict[labels_pl]
    gt = np.argmax(label, axis=-1)
    prediction[gt == 0] = 0
    return compute_label_matrix(gt, prediction, labels_matrix)

def compute_normals_matrix(normals_gt, pred, depth, normals_matrix):
    weights = ~((np.squeeze(depth) == 0) + (normals_gt[..., :] == [0,0,0])[..., 0])
    num_weights = float(np.sum(weights))
    gt_norm = normals_gt / (np.linalg.norm(normals_gt,axis=-1, keepdims=True)+1e-10)
    pred_norm = pred / (np.linalg.norm(pred,axis=-1, keepdims=True)+1e-10)
    cos_dist = (gt_norm[..., 0] * pred_norm[..., 0] + gt_norm[..., 1] * pred_norm[..., 1] + gt_norm[..., 2] * pred_norm[..., 2])
    dist_angle = 180.0 / math.pi * np.arccos(np.clip(cos_dist, -1.0, 1.0))
    masked_dist = dist_angle[weights]
    below_11_25 = np.sum(masked_dist <= 11.25) / num_weights
    below_22_5 = np.sum(masked_dist <= 22.5) / num_weights
    below_30 = np.sum(masked_dist <= 30.0) / num_weights
    mean = np.mean(masked_dist)
    normals_matrix += np.array([below_11_25, below_22_5, below_30, mean])
    return normals_matrix

def get_normals_metrics(normals_matrix, step):
    return normals_matrix / float(step + 1)

def compute_rmse(pred, labels):
    return np.sqrt(((pred - labels) ** 2).mean(axis=-1))

def compute_rel_error(pred, labels):
    return (np.abs(pred - labels) / pred).mean(axis=-1)

def compute_depth_matrix(depth_gt, pred, depth_matrix):
    pred_squeezed = np.squeeze(pred)
    depth_gt_squeezed = np.squeeze(depth_gt)
    weights = ~(depth_gt_squeezed == 0)
    num_weights = float(np.sum(weights))
    pred_mask = pred_squeezed[weights]
    depth_mask = depth_gt_squeezed[weights]
    masked_dist = np.maximum(depth_mask / pred_mask, pred_mask / depth_mask)
    below_1 = np.sum(masked_dist <= 1.25) / num_weights
    below_2 = np.sum(masked_dist <= 1.5625) / num_weights
    below_3 = np.sum(masked_dist <= 1.953124) / num_weights
    rmse_val = compute_rmse(pred_mask, depth_mask) / 1000.0
    rel_error = compute_rel_error(pred_mask, depth_mask)
    depth_matrix += np.array([below_1, below_2, below_3, rmse_val, rel_error])
    return depth_matrix

def get_depth_metrics(depth_matrix, step):
    return depth_matrix / float(step + 1)

def print_info(labels_matrix, normals_matrix, depth_matrix, step, total_num, finished=False):
    normals_metrics = get_normals_metrics(normals_matrix, step)
    depth_metrics = get_depth_metrics(depth_matrix, step)
    if not finished:
        print '%s %s] %d. iou updating' \
          % (str(datetime.datetime.now()), str(os.getpid()), total_num)
    print 'mIoU: ', compute_iou(labels_matrix)
    print '11.25: ', normals_metrics[0], '22.5: ', normals_metrics[1], '30: ', normals_metrics[2], 'mean angle error: ', normals_metrics[3]
    print '1.25: ', depth_metrics[0], '1.25^2: ', depth_metrics[1], '1.25^3: ', depth_metrics[2], 'rmse', depth_metrics[3], 'rel err: ', depth_metrics[4]

def test_func(config):
    module = importlib.import_module('models.' + config['model'])
    model_func = getattr(module, config['model'])
    helper = get_dataset(config)
    modality_infos, num_label_classes = extract_modalities(config)
    data_list, iterator = helper.get_test_data(config, num_label_classes)
    resnet_name = 'resnet_v2_50'

    aux_loss_mode = 'both'
    if 'aux_loss_mode' in config:
        aux_loss_mode = config['aux_loss_mode'].lower()

    with tf.variable_scope(resnet_name):
        model = model_func(modality_infos=modality_infos, aux_loss_mode=aux_loss_mode, training=False)
        images_pl, depths_pl, normals_pl, labels_pl, update_ops = setup_model(model, config, train=False)

    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    sess.run(tf.global_variables_initializer())
    import_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print 'total_variables_loaded:', len(import_variables)
    saver = tf.train.Saver(import_variables)
    saver.restore(sess, config['checkpoint'])
    #sess.run(iterator.initializer)
    step = 0
    total_num = 0
    labels_matrix = np.zeros([num_label_classes, 3])
    normals_matrix = np.zeros([4])
    depth_matrix = np.zeros([5])
    start_step = 0
    # Let's check to see if we have an inference checkpoint
    if 'save_dir' in config.keys():
        try:
            labels_matrix = np.load(config['save_dir'] + '/labels_matrix.npy')
            normals_matrix = np.load(config['save_dir'] + '/normals_matrix.npy')
            depth_matrix = np.load(config['save_dir'] + '/depth_matrix.npy')
            with open(config['save_dir'] + '/output_step.txt', 'r') as f:
                start_step = int(f.read().strip('\n'))
        except:
            print('Could not load save files')

    print('Start step is ', str(start_step), ' mIoU is ', compute_iou(labels_matrix))
    while 1:
        try:
            feed_dict = setup_feeddict(data_list, sess, images_pl, depths_pl, normals_pl, labels_pl, config) 
            if start_step <= step:
                inputs = dict()
                for mod in config['output_modality']:
                    if mod == 'labels':
                        inputs[mod] = model.softmax
                    elif mod == 'normals':
                        inputs[mod] = model.output_normals
                    elif mod == 'depth':
                        inputs[mod] = model.output_depth * 1000
                results = sess.run(list(inputs.values()), feed_dict=feed_dict)
                for mod, result in zip(list(inputs.keys()), results):
                    if mod == 'labels':
                        labels_matrix = get_label_metrics(result, feed_dict, labels_pl, labels_matrix)
                    elif mod == 'normals':
                        normals_matrix = compute_normals_matrix(feed_dict[normals_pl], result, feed_dict[depths_pl], normals_matrix)
                    elif mod == 'depth':
                        depth_matrix = compute_depth_matrix(feed_dict[depths_pl], result, depth_matrix)
                    
                total_num += config['batch_size']
                if (step+1) % config['skip_step'] == 0:
                    print_info(labels_matrix, normals_matrix, depth_matrix, step, total_num, False)

                if 'save_dir' in config.keys() and (step+1) % 1000 == 0:
                    print('Saving evaluation')
                    np.save(config['save_dir'] + '/labels_matrix.npy', labels_matrix)
                    np.save(config['save_dir'] + '/normals_matrix.npy', normals_matrix)
                    np.save(config['save_dir'] + '/depth_matrix.npy', depth_matrix)
                    with open(config['save_dir'] + '/output_step.txt', 'w') as f:
                        f.write(str(step))
            elif (step+1) % 500 == 0:
                print('Skpping step: %d' % (step))
                total_num += config['batch_size']
            else:
                total_num += config['batch_size']
            step += 1

        except tf.errors.OutOfRangeError:
            print_info(labels_matrix, normals_matrix, depth_matrix, step - 1, total_num, True)
            break

def main():
    args = PARSER.parse_args()
    if args.config:
        file_address = open(args.config)
        config = yaml.load(file_address)
    else:
        print '--config config_file_address missing'
    test_func(config)

if __name__ == '__main__':
    main()
