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

import argparse
import cv2
import numpy as np
import tensorflow as tf
import threading
from Queue import Queue

def _int64_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))

def _bytes_feature(data):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-f', '--file')
PARSER.add_argument('-r', '--record')
PARSER.add_argument('-n', '--num_threads')

def decode(txt, read_queue):
    with open(txt) as file_handler:
        all_list = file_handler.readlines()

    file_list = []
    for line in all_list:
        splits = line.strip('\n').split(',')
        read_queue.put((splits[0], splits[1], splits[2]))

def CreateTFExamples(rgb_filename, depth_filename, label_filename):
    rgb = cv2.imread(rgb_filename, cv2.IMREAD_ANYCOLOR)
    depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    label = cv2.imread(label_filename, cv2.IMREAD_ANYDEPTH)
    if depth is None or rgb is None or label is None:
        return None
        
    height = rgb.shape[0]
    width = rgb.shape[1]
    rgb_write = rgb.tostring()
    depth_write = depth.tostring()
    label_write = label.tostring()
    features = {'height':_int64_feature(height),
                'width':_int64_feature(width),
                'rgb':_bytes_feature(rgb_write),
                'depth':_bytes_feature(depth_write),
                'label':_bytes_feature(label_write),
               }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()

def ReadImages(read_queue, write_queue):
    while not read_queue.empty():
        rgb, depth, label = read_queue.get()
        example = CreateTFExamples(rgb, depth, label)
        if example is not None:
            write_queue.put(example)
        read_queue.task_done()
    return True

def WriteTFRecords(record_name, write_queue):
    writer = tf.python_io.TFRecordWriter(record_name)
    count = 0
    while True:
        record = write_queue.get()
        if record == 'Done':
            writer.close()
            write_queue.task_done()
            return
        writer.write(record)
        if (count+1)%5000 == 0:
            print 'Processed data: {}'.format(count)
        write_queue.task_done()
        count += 1

def convert(read_queue, record_name, num_threads):

    write_queue = Queue(maxsize=0)
    worker = threading.Thread(target=WriteTFRecords, kwargs=dict(record_name=record_name, write_queue=write_queue))
    worker.start()
    for i in range(num_threads):
        read_worker = threading.Thread(target=ReadImages, kwargs=dict(read_queue=read_queue, write_queue=write_queue))
        read_worker.start()

    # Wait for read_queue to be empty
    read_queue.join()

    # Send write_queue the finish signal
    write_queue.put('Done')

    # Wait for write_queue to be empty
    write_queue.join()
        

def main():
    args = PARSER.parse_args()
    read_queue = Queue(maxsize=0)
    if args.file:
         decode(args.file, read_queue)
    else:
        print '--file file_address missing'
        return
    if args.record:
        record_name = args.record
    else:
        print '--record tfrecord name missing'
        return
    if args.num_threads:
        num_threads = int(args.num_threads)
    else:
        num_threads = 50
    convert(read_queue, record_name, num_threads)

if __name__ == '__main__':
    main()
