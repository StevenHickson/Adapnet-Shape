import cv2
import numpy as np
import math
import random
from skimage.transform import rotate
from skimage.util import random_noise

def rotate_images(image, depth, normals, label, max_rotation):
    random_degree = random.uniform(-max_rotation, max_rotation)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), random_degree, 1.0)
    new_image = cv2.warpAffine(image, M, (w, h))
    new_depth = cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_NEAREST)
    new_label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST)
    rot_normals = cv2.warpAffine(normals, M, (w, h), flags=cv2.INTER_NEAREST)
    # Recalculate normals
    random_rads = math.radians(random_degree)
    cos_val = math.cos(random_rads)
    sin_val = math.sin(random_rads)
    new_normals = np.copy(rot_normals)
    new_normals[:,:,0] = rot_normals[:,:,0] * cos_val - rot_normals[:,:,1] * sin_val
    new_normals[:,:,1] = rot_normals[:,:,0] * sin_val + rot_normals[:,:,1] * cos_val
    return (new_image, new_depth, new_normals, new_label)
    
def flip_images(image, depth, normals, label, probability):
    op_prob = random.uniform(0,1)
    if op_prob <= probability:
        new_image = image[:, ::-1]
        new_depth = depth[:, ::-1]
        new_normals = normals[:, ::-1]
        new_label = label[:, ::-1]
        # Recalculate normals
        new_normals[:,:,0] *= -1
        return (new_image, new_depth, new_normals, new_label)
    return (image, depth, normals, label)
    
def blur_images(image, depth, normals, label, max_kernel):
    random_kernel = int(np.floor(random.uniform(1, max_kernel)) // 2 * 2 + 1)
    new_image = cv2.GaussianBlur(image,(random_kernel,random_kernel),cv2.BORDER_DEFAULT)
    return (new_image, depth, normals, label)
    
def random_noise_images(image, depth, normals, label, max_variance):
    op_prob = random.uniform(0,max_variance)
    new_image = (random_noise(image, var=op_prob) * 255).astype(np.uint8)
    return (new_image, depth, normals, label)

def crop_images(image, depth, normals, label, max_crop_percent):
    output_shape = depth.shape[::-1]
    crop_width_start = int(np.floor(random.uniform(0, max_crop_percent) * output_shape[0]))
    crop_width = output_shape[0] - int(np.floor(random.uniform(0, max_crop_percent) * output_shape[0]))
    crop_height_start = int(np.floor(random.uniform(0, max_crop_percent) * output_shape[1]))
    crop_height = output_shape[1] - int(np.floor(random.uniform(0, max_crop_percent) * output_shape[1]))
    new_image = cv2.resize(image[crop_height_start:crop_height, crop_width_start:crop_width], output_shape)
    new_depth = cv2.resize(depth[crop_height_start:crop_height, crop_width_start:crop_width], output_shape, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    new_normals = cv2.resize(normals[crop_height_start:crop_height, crop_width_start:crop_width], output_shape, interpolation=cv2.INTER_NEAREST)
    new_label = cv2.resize(label[crop_height_start:crop_height, crop_width_start:crop_width], output_shape, interpolation=cv2.INTER_NEAREST)
    # recalculate depth
    scale = min(float(crop_width - crop_width_start) / output_shape[0], float(crop_height - crop_height_start) / output_shape[1])
    new_depth *= scale
    return (new_image, new_depth.astype(np.uint16), new_normals, new_label)

def augment_images(image, depth, normals, label, config):
    if 'aug_max_rotation' in config:
        image, depth, normals, label = rotate_images(image, depth, normals, label, config['aug_max_rotation'])
    if 'aug_flip_prob' in config:
        image, depth, normals, label = flip_images(image, depth, normals, label, config['aug_flip_prob'])
    if 'aug_blur_max' in config:
        image, depth, normals, label = blur_images(image, depth, normals, label, config['aug_blur_max'])
    if 'aug_max_noise_var' in config:
        image, depth, normals, label = random_noise_images(image, depth, normals, label, config['aug_max_noise_var'])
    if 'aug_max_crop_percent' in config:
        image, depth, normals, label = crop_images(image, depth, normals, label, config['aug_max_crop_percent'])
    return (image, depth, normals, label)
