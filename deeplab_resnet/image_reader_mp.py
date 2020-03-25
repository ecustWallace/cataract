import os

import numpy as np
import tensorflow as tf
import random

def read_labeled_image_list_MP(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    pre_masks = []
    masks = []
    for line in f:
        try:
            image, pre_mask, mask = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            raise Exception("The path is not a triplet.")
        images.append(data_dir + image)
        pre_masks.append((data_dir + pre_mask))
        masks.append(data_dir + mask)
    random.seed(1)
    random.shuffle(images)
    random.seed(1)
    random.shuffle(pre_masks)
    random.seed(1)
    random.shuffle(masks)
    return images, pre_masks, masks


def image_scaling_MP(img, pre_label, label):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """

    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    pre_label = tf.image.resize_nearest_neighbor(tf.expand_dims(pre_label, 0), new_shape)
    pre_label = tf.squeeze(pre_label, squeeze_dims=[0])
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    return img, pre_label, label


def image_mirroring_MP(img, pre_label, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    pre_label = tf.reverse(pre_label, mirror)
    label = tf.reverse(label, mirror)
    return img, pre_label, label


def random_crop_and_pad_image_and_labels_MP(image, pre_label, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label, pre_label = tf.cast(label, dtype=tf.float32), tf.cast(pre_label, dtype=tf.float32)
    label, pre_label = label - ignore_label, pre_label - ignore_label  # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, pre_label, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
                                                tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 5])
    img_crop = combined_crop[:, :, :last_image_dim]
    pre_label_crop = combined_crop[:, :, last_image_dim:last_image_dim+1]
    label_crop = combined_crop[:, :, last_image_dim + 1:]
    label_crop, pre_label_crop = label_crop + ignore_label, pre_label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))
    pre_label_crop.set_shape((crop_h, crop_w, 1))
    return img_crop, pre_label_crop, label_crop


def read_images_from_disk_MP(input_queue, input_size, random_scale, random_mirror, ignore_label,
                             img_mean):  # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.

    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])
    pre_label_contents = tf.read_file(input_queue[1])
    label_contents = tf.read_file(input_queue[2])

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean[:3]
    pre_label = tf.cast(tf.image.decode_png(pre_label_contents, channels=1), dtype=tf.float32)
    label = tf.image.decode_png(label_contents, channels=1)

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, pre_label, label = image_scaling_MP(img, pre_label, label)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, pre_label, label = image_mirroring_MP(img, pre_label, label)

        # Randomly crops the images and labels.
        img, pre_label, label = random_crop_and_pad_image_and_labels_MP(img, pre_label, label, h, w, ignore_label)

    return img, pre_label, label


class ImageReader_MP(object):
    '''
    ImageReader for Mask Propagation Network
    '''

    def __init__(self, data_dir, data_list, input_size,
                 random_scale, random_mirror, ignore_label, img_mean, coord):
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        self.image_list, self.pre_label_list, self.label_list = read_labeled_image_list_MP(self.data_dir,
                                                                                           self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.pre_labels = tf.convert_to_tensor(self.pre_label_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.pre_labels, self.labels],
                                                   shuffle=input_size is not None)
        self.image, self.pre_label, self.label = read_images_from_disk_MP(self.queue, self.input_size, random_scale,
                                                                       random_mirror,
                                                                       ignore_label, img_mean)

    def dequeue(self, num_elements):
        image_batch, pre_label_batch, label_batch = tf.train.batch([self.image, self.pre_label, self.label],
                                                                   num_elements)
        return image_batch, pre_label_batch, label_batch
