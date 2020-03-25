import os
import tensorflow as tf
from PIL import Image
import numpy as np
import re
from tqdm import tqdm

# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

cwd = '/home/deeplab/CNN/mask-propagation-master/train/train_12/'
types = ('Image', 'NoisyImage')
writer = tf.python_io.TFRecordWriter("train_12.tfrecords")

for folder in tqdm(os.listdir(cwd)):
    print(folder)
    if 'NoisyImage' in folder: img_type = 2
    elif 'Image' in folder: img_type = 1
    else: img_type = 0
    if img_type:
        for image in os.listdir(cwd+folder):
            img_path = cwd + folder + '/' + image
            if img_type == 2: label_path = cwd + folder.replace('NoisyImage', 'CNNLabel') + '/' + image.replace('jpg', 'png')
            if img_type == 1: label_path = cwd + folder.replace('Image', 'CNNLabel') + '/' + image.replace('jpg', 'png')
            idx = label_path.rfind('/')
            if label_path[-10] != '/':
                label_path = label_path[:idx+1] + label_path[-9:]

            img = np.asarray(Image.open(img_path)).tobytes()
            label = np.asarray(Image.open(label_path)).tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label":_bytes_feature(label),
                "img":_bytes_feature(img),
                "img_type":_int64_feature(img_type)
            }))
            writer.write(example.SerializeToString())
writer.close()