from tkinter import Image
from matplotlib.transforms import Bbox
import tensorflow as tf
import os

import matplotlib.pyplot as plt
import PIL.Image as Image


def read_info(images_info_path, bboxes_info_path, train_test_split_path):
    images_info = []
    bboxes_info = []
    with open(images_info_path, 'r') as f:
        for line in f:
            line = line.strip()
            image_path = line.split(' ')[1]
            images_info.append(image_path)
    
    with open(bboxes_info_path, 'r') as f:
        for line in f:
            line = line.strip()
            bbox = [float(c) for c in line.split(' ')[1:]]
            bboxes_info.append(bbox)

    train_test_split_info = {'train':[], 'test':[]}
    with open(train_test_split_path, 'r') as f:
        for line in f:
            line = line.strip()
            info = line.split(' ')
            if info[1] == '0':
                train_test_split_info['test'].append(int(info[0]))
            else:
                train_test_split_info['train'].append(int(info[0]))

    return images_info, bboxes_info, train_test_split_info

def round_to_int(float_value):
    return tf.cast(tf.math.round(float_value), dtype=tf.int32)

def get_preprocess_fuc(padding, image_size):

    def preprocess_image(image_path, bbox):
        # unnormalize bounding box coordinates
        image = tf.io.read_file(image_path)
        
        image = tf.image.decode_jpeg(image, channels=3)
        x1 = bbox[1]
        y1 = bbox[0]
        x2 = bbox[3] + x1
        y2 = bbox[2] + y1

        height = tf.cast(tf.shape(image)[0], dtype=tf.float32)
        width = tf.cast(tf.shape(image)[1], dtype=tf.float32)
        bounding_box = tf.stack([x1, y1, x2, y2])

        # calculate center and length of longer side, add padding
        target_center_y = 0.5 * (bounding_box[0] + bounding_box[2])
        target_center_x = 0.5 * (bounding_box[1] + bounding_box[3])
        target_size = tf.maximum(
            (1.0 + padding) * (bounding_box[2] - bounding_box[0]),
            (1.0 + padding) * (bounding_box[3] - bounding_box[1]),
        )

        # modify crop size to fit into image
        target_height = tf.reduce_min(
            [target_size, 2.0 * target_center_y, 2.0 * (height - target_center_y)]
        )
        target_width = tf.reduce_min(
            [target_size, 2.0 * target_center_x, 2.0 * (width - target_center_x)]
        )

        # crop image
        image = tf.image.crop_to_bounding_box(
            image,
            offset_height=round_to_int(target_center_y - 0.5 * target_height),
            offset_width=round_to_int(target_center_x - 0.5 * target_width),
            target_height=round_to_int(target_height),
            target_width=round_to_int(target_width),
        )

        # resize and clip
        # for image downsampling, area interpolation is the preferred method
        image = tf.image.resize(
            image, size=[image_size, image_size], method=tf.image.ResizeMethod.AREA
        )
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)
    
    return preprocess_image

def prepare_dataset(image_dir, images_info_path, bboxes_info_path, train_test_split_path, split, batch_size, padding, image_size):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID calculation
    images_info, bboxes_info, train_test_split_info = read_info(images_info_path, bboxes_info_path, train_test_split_path)
    if split == 'train':
        images_info = [os.path.join(image_dir,images_info[i-1]) for i in train_test_split_info['train']]
        bboxes_info = [bboxes_info[i-1] for i in train_test_split_info['train']]
    else:
        images_info = [os.path.join(image_dir,images_info[i-1]) for i in train_test_split_info['test']]
        bboxes_info = [bboxes_info[i-1] for i in train_test_split_info['test']]

    dataset = tf.data.Dataset.from_tensor_slices((images_info, bboxes_info))

    prepare_fuc = get_preprocess_fuc(padding, image_size)

    return (
        dataset.map(prepare_fuc, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
