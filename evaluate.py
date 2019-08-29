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
from dataset.helper import DatasetHelper, compute_output_matrix, compute_iou
from train_utils import *

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-c', '--config', default='config/cityscapes_test.config')

def test_func(config):
    module = importlib.import_module('models.' + config['model'])
    model_func = getattr(module, config['model'])
    helper = DatasetHelper()
    helper.Setup(config)
    modalities_num_classes, num_label_classes = extract_modalities(config)
    data_list, iterator = helper.get_test_data(config, num_label_classes)
    resnet_name = 'resnet_v2_50'

    with tf.variable_scope(resnet_name):
        model = model_func(modalities_num_classes=modalities_num_classes, training=False)
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
    output_matrix = np.zeros([num_label_classes, 3])
    start_step = 0
    # Let's check to see if we have an inference checkpoint
    if 'save_dir' in config.keys():
        try:
            output_matrix = np.load(config['save_dir'] + '/output_matrix.npy')
            with open(config['save_dir'] + '/output_step.txt', 'r') as f:
                start_step = int(f.read().strip('\n'))
        except:
            print('Could not load save files')

    print('Start step is ', str(start_step), ' mIoU is ', compute_iou(output_matrix))
    while 1:
        try:
            feed_dict = setup_feeddict(data_list, sess, images_pl, depths_pl, normals_pl, labels_pl, config) 
            if start_step <= step:
                probabilities = sess.run([model.softmax], feed_dict=feed_dict)
                prediction = np.argmax(probabilities[0], 3)
                label = feed_dict[labels_pl]
                gt = np.argmax(label, 3)
                prediction[gt == 0] = 0
                output_matrix = compute_output_matrix(gt, prediction, output_matrix)
                total_num += label.shape[0]
                if (step+1) % config['skip_step'] == 0:
                    print '%s %s] %d. iou updating' \
                      % (str(datetime.datetime.now()), str(os.getpid()), total_num)
                    print 'mIoU: ', compute_iou(output_matrix)

                if 'save_dir' in config.keys() and (step+1) % 1000 == 0:
                    print('Saving evaluation')
                    np.save(config['save_dir'] + '/output_matrix.npy', output_matrix)
                    with open(config['save_dir'] + '/output_step.txt', 'w') as f:
                        f.write(str(step))
            elif (step+1) % 500 == 0:
                print('Skpping step: %d' % (step))
                total_num += config['batch_size']
            else:
                total_num += config['batch_size']
            step += 1

        except tf.errors.OutOfRangeError:
            print 'mIoU: ', compute_iou(output_matrix), 'total_data: ', total_num
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
