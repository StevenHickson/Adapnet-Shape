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

import tensorflow as tf
from AdapNet_base import AdapNet_base

class AdapNet_fused(AdapNet_base):
    def setup(self, data):   
        self.input_shape = data.get_shape()

        self.eAspp_out = self.build_encoder(data)

        ### Upsample/Decoder
        deconv_up1, deconv_up2, up2 = self.build_decoder()
        self.total_num_classes = 0
        split_sizes = []
        for modality, num_classes, _ in self.modality_infos: 
            self.total_num_classes += num_classes
            split_sizes.append(num_classes)

        with tf.variable_scope('decoder'):
            up2 = self.conv_batchN_relu(up2, 1, 1, self.total_num_classes, name='conv78')
            deconv_up3 = self.tconv2d(up2, 8, self.total_num_classes, 4)
            deconv_up3 = self.batch_norm(deconv_up3)      
        ## Auxilary
            if self.aux_loss_mode != 'false':
                aux1 = tf.image.resize_images(self.conv_batchN_relu(deconv_up2, 1, 1, self.total_num_classes, name='conv911', relu=False), [self.input_shape[1], self.input_shape[2]])
                aux2 = tf.image.resize_images(self.conv_batchN_relu(deconv_up1, 1, 1, self.total_num_classes, name='conv912', relu=False), [self.input_shape[1], self.input_shape[2]])

        split_id = 0
        splits = tf.split(deconv_up3, split_sizes, axis=-1)
        aux1_splits = tf.split(aux1, split_sizes, axis=-1)
        aux2_splits = tf.split(aux2, split_sizes, axis=-1)
        for modality, num_classes,_ in self.modality_infos: 
            if modality == 'labels':
                self.softmax = tf.nn.softmax(splits[split_id])
                self.output_labels = self.softmax
                if self.aux_loss_mode in [modality, 'both', 'true']:
                    self.aux1_labels = tf.nn.softmax(aux1_splits[split_id])
                    self.aux2_labels = tf.nn.softmax(aux2_splits[split_id])
            elif modality == 'normals':
                self.output_normals = splits[split_id]
                if self.aux_loss_mode in [modality, 'both', 'true']:
                    self.aux1_normals = aux1_splits[split_id]
                    self.aux2_normals = aux2_splits[split_id]
            split_id += 1
        
def main():
    print 'Do Nothing'
   
if __name__ == '__main__':
    main()

