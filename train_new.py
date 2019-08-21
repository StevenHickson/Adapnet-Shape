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
import re
import tensorflow as tf
import yaml
from dataset.helper import DatasetHelper
from train_utils import *

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-c', '--config', default='config/cityscapes_train.config')

def calculate_weights(depths, normals):
    valid_depths = tf.math.not_equal(tf.cast(depths, tf.float32), 0)
    valid_normals = tf.expand_dims(tf.math.not_equal(tf.reduce_sum(tf.math.abs(normals), axis=-1), 0), axis=-1)
    return tf.cast(tf.math.logical_and(valid_depths, valid_normals), tf.float32)

def setup_model(model, config):
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
    return images_pl, depths_pl, normals_pl, labels_pl, update_ops

def original_restore(sess, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    var_str = reader.debug_string()
    name_var = re.findall('[A-Za-z0-9/:_]+ ', var_str)
    import_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    initialize_variables = {} 
    for var in import_variables:
        if var.name+' ' in  name_var:
            initialize_variables[var.name] = var

    saver = tf.train.Saver(initialize_variables, reshape=True)
    saver.restore(save_path=save_file, sess=sess)

def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars, reshape=True)
    saver.restore(session, save_file)

def train_func(config):
    #os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
    module = importlib.import_module('models.'+config['model'])
    model_func = getattr(module, config['model'])
    helper = DatasetHelper()
    helper.Setup(config)
    modalities_num_classes = dict()
    num_label_classes = None
    for modality, num_classes in zip(config['output_modality'], config['num_classes']):
        modalities_num_classes[modality] = num_classes
        if modality == 'labels':
            num_label_classes = num_classes
    data_list, iterator = helper.get_train_data(config, num_label_classes)
    resnet_name = 'resnet_v2_50'
    global_step = tf.train.get_or_create_global_step()
    step = 0

    with tf.variable_scope(resnet_name):
        model = model_func(modalities_num_classes=modalities_num_classes, learning_rate=config['learning_rate'],
                           decay_steps=config['max_iteration'], power=config['power'],
                           global_step=global_step)
        images_pl, depths_pl, normals_pl, labels_pl, update_ops = setup_model(model, config)
 
    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    writer = tf.summary.FileWriter(config['summary_dir'], sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    total_loss = 0.0
    t0 = None
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(config['checkpoint'],
                                                                      'checkpoint')))
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver(max_to_keep=1000)
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])+1
        sess.run(tf.assign(global_step, step))
        print 'Model Loaded'

    else:
        if 'orig_intialize' in config:
            original_restore(sess, config['orig_intialize'])
            print 'Original Pretrained Intialization'
        elif 'intialize' in config:
            optimistic_restore(sess, config['intialize'])
            print 'Optimistic Pretrained Intialization'
        saver = tf.train.Saver(max_to_keep=1000)
       
    while 1:
        try:
            input_names_to_feeds = dict()
            if config['input_modality'] == 'rgb':
                input_names_to_feeds['rgb'] = data_list[0]
            elif config['input_modality'] == 'depth':
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
                    
            inputs = [model.loss, model.train_op, model.summary_op] + update_ops
            result = sess.run(inputs, feed_dict=feed_dict)
            loss_batch = result[0]
            summary = result[2]
            if (step + 1) % config['summaries_step'] == 0:
                writer.add_summary(summary, global_step=step)
            total_loss += loss_batch

            if (step + 1) % config['save_step'] == 0:
                saver.save(sess, os.path.join(config['checkpoint'], 'model.ckpt'), step)

            if (step + 1) % config['skip_step'] == 0:
                left_hours = 0

                if t0 is not None:
                    delta_t = (datetime.datetime.now() - t0).seconds
                    left_time = (config['max_iteration'] - step) / config['skip_step'] * delta_t
                    left_hours = left_time/3600.0

                t0 = datetime.datetime.now()
                total_loss /= config['skip_step']
                print '%s %s] Step %s, lr = %f ' \
                  % (str(datetime.datetime.now()), str(os.getpid()), step,
                     model.lr.eval(session=sess))
                print '\t loss = %.4f' % (total_loss)
                print '\t estimated time left: %.1f hours. %d/%d' % (left_hours, step,
                                                                     config['max_iteration'])
                print '\t', config['model']
                total_loss = 0.0

            step += 1
            if step > config['max_iteration']:
                saver.save(sess, os.path.join(config['checkpoint'], 'model.ckpt'), step-1)
                print 'training_completed'
                break

        except tf.errors.OutOfRangeError:
            print 'Epochs in dataset repeat < max_iteration'
            break

def main():
    args = PARSER.parse_args()
    if args.config:
        file_address = open(args.config)
        config = yaml.load(file_address)
    else:
        print '--config config_file_address missing'
    train_func(config)

if __name__ == '__main__':
    main()
