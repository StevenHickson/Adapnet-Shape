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

class AdapNet_pp(AdapNet_base):
    def setup(self, data):
        # There should only be one key, value in this dict.
        self.modality, self.num_classes, _ = modality_infos[0]
        self.input_shape = data.get_shape()

        self.eAspp_out = self.build_encoder(data)

        ### Upsample/Decoder
        deconv_up1, deconv_up2, up2 = self.build_decoder()
        with tf.variable_scope('conv5'):
            self.deconv_up3 = self.tconv2d(up2, 8, self.num_classes, 4)
            self.deconv_up3 = self.batch_norm(self.deconv_up3)      

        if self.modality == 'normals':
            self.output_normals = self.deconv_up3
        else:
            self.softmax = tf.nn.softmax(self.deconv_up3)
            self.output_labels = self.softmax
        ## Auxilary
        if self.aux_loss_mode in [self.modality, 'both', 'true']:
            self.aux1 = tf.image.resize_images(self.conv_batchN_relu(self.deconv_up2, 1, 1, self.num_classes, name='conv911', relu=False), [self.input_shape[1], self.input_shape[2]])
            self.aux2 = tf.image.resize_images(self.conv_batchN_relu(self.deconv_up1, 1, 1, self.num_classes, name='conv912', relu=False), [self.input_shape[1], self.input_shape[2]])
            if self.modality == 'labels':
                self.aux1 = tf.nn.softmax(self.aux1)
                self.aux2 = tf.nn.softmax(self.aux2)
        
def main():
    print 'Do Nothing'
   
if __name__ == '__main__':
    main()

