''' AdapNet++:  Self-Supervised Model Adaptation for Multimodal Semantic Segmentation

 Copyright (C) 2018  Abhinav Valada, Rohit Mohan and Wolfram Burgard

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.'''



import tensorflow as tf
from dataset.helper import *
import numpy as np
import argparse
import os
import datetime
import importlib
import yaml
import re
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('-c','--config')

def train_func(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
    module = importlib.import_module('models.'+config['model'])
    model_func = getattr(module,config['model'])     
    data_list, iterator = get_train_data(config)
    resnet_name='resnet_v1_50'
    global_step=tf.Variable(0,trainable=False,name='Global_Step')
   
    with tf.variable_scope(resnet_name):
    	model = model_func(num_classes=config['num_classes'],learning_rate=config['learning_rate'],decay_steps=config['max_iteration'],power=config['power'],global_step=global_step)
        images_pl = tf.placeholder(tf.float32, [None, config['height'],config['width'], 3])
        labels_pl = tf.placeholder(tf.float32, [None, config['height'],config['width'], config['num_classes']])
        logits = model.build_graph(images_pl,labels_pl)
    config1 =  tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess=tf.Session(config=config1)
    sess.run(tf.global_variables_initializer())
    step=0
    total_loss=0.0
    t0 = None
    import_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print 'total_variables_loaded:',len(import_variables)        
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(config['checkpoint'],'checkpoint')))
    if ckpt and ckpt.model_checkpoint_path:
            model.create_optimizer()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1000)
            saver.restore(sess, ckpt.model_checkpoint_path)
            step=int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])+1
            sess.run(tf.assign(global_step,step))
            print 'Model Loaded'

    else: 
       initialize_variables={}
       var_list=['conv1/','block']
       for var in import_variables: 
           if any(elem in var.name for elem in var_list):
              initialize_variables[var.name]=var
              
       #print len(initialize_variables)
       saver=tf.train.Saver(initialize_variables)
       saver.restore(save_path=config['intialize'],sess=sess)
       saver=tf.train.Saver(max_to_keep=1000)
       model.create_optimizer()  
       sess.run(tf.global_variables_initializer())
    
       print 'Intialized'
    
    
     
    
    while 1:
       try: 
           img,label=sess.run([data_list[0],data_list[1]]) 
           
                
           feed_dict={images_pl: img, labels_pl: label}
           loss_batch, _= sess.run([model.loss, model.train_op], 
                                   feed_dict=feed_dict)

           total_loss += loss_batch
             
           if (step + 1) % config['save_step'] == 0:
              saver.save(sess, os.path.join(config['checkpoint'],'model.ckpt'), step)
                   
           if (step + 1) % config['skip_step'] == 0:
              left_hours = 0

              if t0 is not None:
                 delta_t = (datetime.datetime.now() - t0).seconds
                 left_time = (config['max_iteration'] - step) / config['skip_step'] * delta_t
                 left_hours = left_time/3600.0

              t0 = datetime.datetime.now()

              total_loss /= config['skip_step']

              print '%s %s] Step %s, lr = %f ' \
                  % (str(datetime.datetime.now()), str(os.getpid()), step, model.lr.eval(session=sess))
              print '\t loss = %.4f' % (total_loss)
              
              print '\t estimated time left: %.1f hours. %d/%d' % (left_hours, step, config['max_iteration'])
              print '\t',config['model']
              total_loss = 0.0
           step += 1
           if step>config['max_iteration']:
              saver.save(sess, os.path.join(config['checkpoint'],'model.ckpt'), step-1)
              print 'training_completed'
              break
              
       except tf.errors.OutOfRangeError:
                print 'Epochs in dataset repeat < max_iteration'
                break     











def main():
    args = parser.parse_args()
    if args.config:
        f=open(args.config)
        config=yaml.load(f)
    else:
        print '--config config_file_address missing'
    train_func(config)

if __name__=='__main__':
        main()
