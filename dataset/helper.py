''' AdapNet:  Adaptive  Semantic  Segmentation
              in  Adverse  Environmental  Conditions

 Copyright (C) 2018  Abhinav Valada, Johan Vertens , Ankit Dhall and Wolfram Burgard

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.'''

import cv2
import csv
import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append('/nethome/shickson3/CreateNormals/')
from python.calc_normals import NormalCalculation
from augmentation import augment_images

output_width = None
output_height = None

def parser(image_decoded, depth_decoded, normals_decoded, label_decoded, num_label_classes):
    image_decoded.set_shape([None, None, None])
    depth_decoded.set_shape([None, None, None])
    normals_decoded.set_shape([None, None, None])
    #depth_decoded.set_shape([None, None, None])
    label_decoded.set_shape([None, None, None])
    #image_resized = tf.image.resize(image_decoded, [self.config['width'], self.config['height']])
    #depth_resized = tf.cast(tf.image.resize(depth_decoded, [self.config['width'], self.config['height']], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), tf.int32)
    #label_resized = tf.cast(tf.image.resize(label_decoded, [self.config['height'], self.config['width']], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), tf.int32)
    label = tf.cast(label_decoded, tf.int32)
    depth = tf.cast(depth_decoded, tf.float32)
    normals = tf.cast(normals_decoded, tf.float32)

    normals = tf.reshape(normals, [output_height, output_width, 3])
    depth = tf.reshape(depth, [output_height, output_width, 1])
    label = tf.reshape(label, [output_height, output_width, 1])
    label = tf.one_hot(label, num_label_classes)
    label = tf.squeeze(label, axis=2)
    modality1 = tf.cast(tf.reshape(image_decoded, [output_height, output_width, 3]), tf.float32)
    #modality1 = (modality1 - 127.5) / 2.0

    return modality1, depth, normals, label

class DatasetHelper:

    normal_calculator = None

    def MapLabels(self, label):
        raise NotImplementedError()

    def Setup(self, config):
        global output_width
        global output_height

        self.config = config
        self.name = config['dataset_name']
	output_width = config['width']
	output_height = config['height']

    def read_raw_images(self, image_file, depth_file, label_file, num_label_classes):
        image_decoded = tf.io.read_file(image_file)
        depth_decoded = tf.io.read_file(depth_file)
        label_decoded = tf.io.read_file(label_file)
        return image_decoded, depth_decoded, label_decoded

    def _read_images_function(self, image_file, depth_file, label_file, num_label_classes, dataset_name, compute_normals):
        image_decoded = cv2.imread(image_file.decode(), cv2.IMREAD_COLOR)
        depth_decoded = cv2.imread(depth_file.decode(), cv2.IMREAD_ANYDEPTH)
        label_decoded = cv2.imread(label_file.decode(), cv2.IMREAD_ANYDEPTH)
        if image_decoded is None or label_decoded is None or depth_decoded is None:
            image_decoded = np.zeros((self.config['height'], self.config['width'], 3), dtype=np.uint8)
            label_decoded = np.zeros((self.config['height'], self.config['width']), dtype=np.uint16)
            depth_decoded = np.zeros((self.config['height'], self.config['width']), dtype=np.uint16)
            normals_decoded = np.zeros((self.config['height'], self.config['width'], 3), dtype=np.float32)
        elif compute_normals:
            resized_labels = cv2.resize(label_decoded, depth_decoded.shape[::-1], interpolation=cv2.INTER_NEAREST)
            normals_decoded = self.normal_calculator.Calculate(depth_decoded, resized_labels)
        else:
            normals_decoded = np.zeros_like(image_decoded).astype(np.float32)
        image_decoded = cv2.cvtColor(cv2.resize(image_decoded, (self.config['width'],self.config['height'])), cv2.COLOR_BGR2RGB)
        depth_decoded = cv2.resize(depth_decoded, (self.config['width'],self.config['height']), interpolation=cv2.INTER_NEAREST)
        label_decoded = cv2.resize(label_decoded, (self.config['width'],self.config['height']), interpolation=cv2.INTER_NEAREST)
        normals_decoded = cv2.resize(normals_decoded, (self.config['width'],self.config['height']), interpolation=cv2.INTER_NEAREST)
        label_decoded = self.MapLabels(label_decoded)

        # Augment the images if the parameters are in the config file.
        image_decoded, depth_decoded, normals_decoded, label_decoded = augment_images(image_decoded, depth_decoded, normals_decoded, label_decoded, self.config)

        return image_decoded, depth_decoded, normals_decoded, label_decoded, num_label_classes

    def get_batch(self, split, config, num_label_classes):
        filenames = config[split]
        compute_normals = ('normals' in self.config['output_modality'] or self.config['input_modality'] == 'normals')
        image_files = []
        depth_files = []
        label_files = []
        with open(filenames, 'r') as text_file:
            for line in text_file:
                splits = line.strip('\n').split(',')
                image_files.append(splits[0])
                depth_files.append(splits[1])
                label_files.append(splits[2])

        dataset = tf.data.Dataset.from_tensor_slices((image_files, depth_files, label_files))
        #dataset = dataset.map(lambda image_file, depth_file, label_file: read_raw_images(image_file, depth_file, label_file, config['num_label_classes']))
        dataset = dataset.map(
        lambda image_file, depth_file, label_file: tuple(tf.py_func(
            self._read_images_function, [image_file, depth_file, label_file, num_label_classes, self.name, compute_normals], [tf.uint8, tf.uint16, tf.float32, tf.uint16, tf.int32])))
        dataset = dataset.map(parser, num_parallel_calls=8)
        return dataset

    def get_train_data(self, config, num_label_classes):
        dataset = self.get_batch('train_data', config, num_label_classes)
        if 'buffer_size' in config:
            num_buffer = config['buffer_size']
        else:
            num_buffer = 200
        dataset = dataset.shuffle(buffer_size=num_buffer)
        dataset = dataset.batch(config['batch_size'])
        dataset = dataset.repeat(num_buffer)
        dataset = dataset.prefetch(32)
        iterator = dataset.make_one_shot_iterator()
        rgb, depth, normals, label = iterator.get_next()
        return [rgb, depth, normals, label], iterator

    def get_test_data(self, config, num_label_classes):
        dataset = self.get_batch('test_data', config, num_label_classes)
        dataset = dataset.batch(config['batch_size'])
        dataset = dataset.prefetch(8)
        iterator = dataset.make_one_shot_iterator()
        rgb, depth, normals, label = iterator.get_next()
        return [rgb, depth, normals, label], iterator

