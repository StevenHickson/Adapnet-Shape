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
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
sys.path.append('/nethome/shickson3/CreateNormals/')
from python.calc_normals import NormalCalculation

normal_calculator = None
output_width = None
output_height = None
camera_params = []
normal_params = []
flat_labels = []
label_nyu_mapping = dict()

def SetUpNormalCalculation():
    global normal_calculator
    normal_calculator = NormalCalculation(camera_params, normal_params, flat_labels)
    
def CreateScenenetMapping(num_classes):
    global camera_params
    global normal_params
    global label_nyu_mapping

    camera_params = [277.128129211,0,160,0,289.705627485,120,0,0,1]
    normal_params = [5,0.02,10]
    for i in range(0, num_classes):
        label_nyu_mapping[i] = i

def MapLabels(label, name):
    if name == 'scannet':
        label_nyu = np.array([label_nyu_mapping[x] for x in label.flatten()])
        return label_nyu.reshape(label.shape).astype(np.uint16)
    return label.astype(np.uint16)

def CreateScannetMapping():
    global camera_params
    global normal_params
    global flat_labels
    global label_nyu_mapping

    camera_params = [577.591,0,318.905,0,578.73,242.684,0,0,1]
    normal_params = [5,0.02,30]
    flat_labels = [1,2,7,8,9,11,19,22,24,29,30,32]

    convert_to_20 = [0,1,2,3,4,5,6,7,8,9,10,11,12,9,13,20,14,20,4,20,2,0,0,0,15,20,0,0,16,0,20,0,20,17,18,20,19,0,20,20,0]
    label_nyu_mapping[0] = 0
    with open('/data4/scannetv2-labels.combined.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        start = True
        for row in reader:
            if not start:
                label_nyu_mapping[int(row[0])] = convert_to_20[int(row[4])]
            start = False

def parser(image_decoded, depth_decoded, normals_decoded, label_decoded, num_classes):
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
    label = tf.one_hot(label, num_classes)
    label = tf.squeeze(label, axis=2)
    modality1 = tf.cast(tf.reshape(image_decoded, [output_height, output_width, 3]), tf.float32)
    modality1 = (modality1 - 127.5) / 2.0

    return modality1, depth, normals, label

class DatasetHelper:

    def Setup(self, config):
        global output_width
        global output_height

        self.config = config
        self.name = config['dataset_name']
	output_width = config['width']
	output_height = config['height']
        if self.name == 'scannet':
	    CreateScannetMapping()
	elif self.name == 'scenenet':
	    CreateScenenetMapping(config['num_classes'])
        if self.config['output_modality'] == 'normals' or self.config['input_modality'] == 'normals':
            SetUpNormalCalculation()

    def read_raw_images(self, image_file, depth_file, label_file, num_classes):
        image_decoded = tf.io.read_file(image_file)
        depth_decoded = tf.io.read_file(depth_file)
        label_decoded = tf.io.read_file(label_file)
        return image_decoded, depth_decoded, label_decoded

    def _read_images_function(self, image_file, depth_file, label_file, num_classes, dataset_name, compute_normals):
        image_decoded = cv2.imread(image_file.decode(), cv2.IMREAD_COLOR)
        depth_decoded = cv2.imread(depth_file.decode(), cv2.IMREAD_ANYDEPTH)
        label_decoded = cv2.imread(label_file.decode(), cv2.IMREAD_ANYDEPTH)
        if compute_normals:
            normals_decoded = normal_calculator.Calculate(depth_decoded, cv2.resize(label_decoded, depth_decoded.shape, interpolation=cv2.INTER_NEAREST))
        else:
            normals_decoded = np.zeros_like(image_decoded).astype(np.float32)
        image_decoded = cv2.resize(image_decoded, (self.config['width'],self.config['height']))
        depth_decoded = cv2.resize(depth_decoded, (self.config['width'],self.config['height']), interpolation=cv2.INTER_NEAREST)
        label_decoded = cv2.resize(label_decoded, (self.config['width'],self.config['height']), interpolation=cv2.INTER_NEAREST)
        normals_decoded = cv2.resize(normals_decoded, (self.config['width'],self.config['height']), interpolation=cv2.INTER_NEAREST)
        label_decoded = MapLabels(label_decoded, dataset_name)
        return image_decoded, depth_decoded, normals_decoded, label_decoded, num_classes

    def get_train_batch(self, config):
        filenames = config['train_data']
        compute_normals = (self.config['output_modality'] == 'normals' or self.config['input_modality'] == 'normals')
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
        #dataset = dataset.map(lambda image_file, depth_file, label_file: read_raw_images(image_file, depth_file, label_file, config['num_classes']))
        dataset = dataset.map(
        lambda image_file, depth_file, label_file: tuple(tf.py_func(
            self._read_images_function, [image_file, depth_file, label_file, config['num_classes'], self.name, compute_normals], [tf.uint8, tf.uint16, tf.float32, tf.uint16, tf.int32])))
        dataset = dataset.map(parser)
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(config['batch_size'])
        dataset = dataset.repeat(100)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def get_train_data(self, config):
        iterator = self.get_train_batch(config)
        rgb, depth, normals, label = iterator.get_next()
        return [rgb, depth, normals, label], iterator

    def get_test_data(self, config):
        iterator = self.get_test_batch(config)
        rgb, depth, normals, label = iterator.get_next()
        return [rgb, depth, normals, label], iterator

    def get_test_batch(self, config):
        filenames = [config['test_data']]
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(lambda x: parser(x, config['num_classes']))
        dataset = dataset.batch(config['batch_size'])
        iterator = dataset.make_initializable_iterator()
        return iterator

    def compute_output_matrix(label_max, pred_max, output_matrix):
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

