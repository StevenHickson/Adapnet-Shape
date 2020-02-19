from helper import DatasetHelper
import numpy as np
import cv2
from augmentation import augment_images

label_mapping = [0,0,0,0,0,0,0,1,2,0,0,3,4,5,0,0,0,6,0,7,8,9,10,11,12,13,14,15,16,0,0,17,18,19,0]

class Cityscape19Dataset(DatasetHelper):

    def MapLabels(self, label):
        label_cityscapes = np.array([label_mapping[x] for x in label.flatten()])
        return label_cityscapes.reshape(label.shape).astype(np.uint16)

    def _read_images_function(self, image_file, depth_file, label_file, num_label_classes, dataset_name, compute_normals):
        depth_file_name = depth_file.decode()
        image_decoded = cv2.imread(image_file.decode(), cv2.IMREAD_COLOR)
        depth_decoded = cv2.imread(depth_file_name, cv2.IMREAD_ANYDEPTH)
        label_decoded = cv2.imread(label_file.decode(), cv2.IMREAD_ANYDEPTH)
        normals_decoded = None
        if image_decoded is None or label_decoded is None or depth_decoded is None:
            image_decoded = np.zeros((self.config['height'], self.config['width'], 3), dtype=np.uint8)
            label_decoded = np.zeros((self.config['height'], self.config['width']), dtype=np.uint16)
            depth_decoded = np.zeros((self.config['height'], self.config['width']), dtype=np.uint16)
            normals_decoded = np.zeros((self.config['height'], self.config['width'], 3), dtype=np.float32)
        elif compute_normals != 0:
            normals_file_name = depth_file_name.replace('disparity','normals').replace('.png','.exr')
            normals_decoded = cv2.imread(normals_file_name, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            if normals_decoded is None:
                normals_decoded = np.zeros_like(image_decoded).astype(np.float32)
                print('Normals file does not exist')
            else:
                normals_decoded = normals_decoded.astype(np.float32)
        else:
            normals_decoded = np.zeros_like(image_decoded).astype(np.float32)
        image_decoded = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)
        label_decoded = self.MapLabels(label_decoded)

        # Augment the images if the parameters are in the config file.
        image_decoded, depth_decoded, normals_decoded, label_decoded = augment_images(image_decoded, depth_decoded, normals_decoded, label_decoded, self.config)

        image_decoded = cv2.resize(image_decoded, (self.config['width'],self.config['height']))
        depth_decoded = cv2.resize(depth_decoded, (self.config['width'],self.config['height']), interpolation=cv2.INTER_NEAREST)
        label_decoded = cv2.resize(label_decoded, (self.config['width'],self.config['height']), interpolation=cv2.INTER_NEAREST)
        normals_decoded = cv2.resize(normals_decoded, (self.config['width'],self.config['height']), interpolation=cv2.INTER_NEAREST)


        return image_decoded, depth_decoded, normals_decoded, label_decoded, num_label_classes
