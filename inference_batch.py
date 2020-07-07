"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

IMG_MEAN = np.array((157.9416,111.3888,85.2623), dtype=np.float32)

NUM_CLASSES = 3
SAVE_DIR = '/home/deeplab/CNN/mask-propagation-master/inference/output/train_12+_MP/N3_600_C/'
DATA_DIRECTORY = '/home/deeplab/CNN/mask-propagation-master/inference/dataset'
DATA_LIST_PATH = '/home/deeplab/CNN/mask-propagation-master/inference/N3_600_C.txt'
NUM_STEPS = 538
IGNORE_LABEL = 255
WEIGHT='/home/deeplab/CNN/mask-propagation-master/snapshots/train_12+_MP/model.ckpt-90000'

# START_LABEL = '/home/deeplab/CNN/mask-propagation-master/N7_start.png'
START_LABEL = None


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--model-weights", type=str, default=WEIGHT,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    # #os.environ['CUDA_VISIBLE_DEVICES']='2'
    # """Create the model and start the evaluation process."""
    # args = get_arguments()
    #
    # # Create queue coordinator.
    # coord = tf.train.Coordinator()
    #
    # # Load reader.
    # with tf.name_scope("create_inputs"):
    #     reader = ImageReader(
    #                          args.data_dir,
    #                          args.data_list,
    #                          None, # No defined input size.
    #                          False, # No random scale.
    #                          False, # No random mirror.
    #                          args.ignore_label,
    #                          IMG_MEAN,
    #                          coord)
    #     image, label = reader.image, reader.label
    # image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.
    #
    # # Create network.
    # net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)
    #
    # # Which variables to load.
    # restore_var = tf.global_variables()
    #
    # # Predictions.
    # raw_output = net.layers['fc1_voc12']
    # raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    # raw_output_up = tf.argmax(raw_output_up, dimension=3)
    # pred = tf.expand_dims(raw_output_up, dim=3)
    #
    #
    # # Set up TF session and initialize variables.
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # init = tf.global_variables_initializer()
    #
    # sess.run(init)
    #
    # # Load weights.
    # loader = tf.train.Saver(var_list=restore_var)
    # load(loader, sess, args.model_weights)
    #
    # # Start queue threads.
    # threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    #
    # count=1
    # for step in range(args.num_steps):
    # # Perform inference.
    #     preds = sess.run(pred)
    #     msk = decode_labels(preds, num_classes=args.num_classes)
    #     im = Image.fromarray(msk[0])
    #     if not os.path.exists(args.save_dir):
    #         os.makedirs(args.save_dir)
    #     im.save(args.save_dir +str(count)+'.png')
    #
    #     print('The output file has been saved to {}'.format(args.save_dir + str(count)+'mask.png'))
    #     count+=1

    # os.environ['CUDA_VISIBLE_DEVICES']='2'
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None,  # No defined input size.
            False,  # No random scale.
            False,  # No random mirror.
            args.ignore_label,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)  # Add one batch dimension.

    # Make the first image batch
    # pre_label = tf.expand_dims(tf.cast(tf.image.decode_png(tf.convert_to_tensor(START_LABEL + '1.png', dtype=tf.string), channels=1), dtype=tf.float32), dim=0)
    if START_LABEL is None:
        pre_label = tf.zeros([1,600,600,1], dtype=tf.float32)
    else:
        content = tf.read_file(START_LABEL)
        pre_label = tf.cast(tf.expand_dims(tf.image.decode_png(content, channels=1), dim=0), dtype=tf.float32)
    # pre_label = tf.zeros([1,600,600,1], dtype=tf.float32)
    # image_batch_pre = tf.concat([image_batch, pre_label], axis=3)

    # Create network.
    net = DeepLabResNetModel({'data': image_batch, 'pre_label':pre_label}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)

    # Start queue threads
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    count = 1
    for step in range(args.num_steps):
        a = time.time()
        # Perform inference.
        if step == 0:
            preds = sess.run(pred)
        else:
            preds = sess.run(pred, feed_dict={pre_label:preds})
            # preds = sess.run(pred, feed_dict={path:START_LABEL})
        # pre_label_val = tf.convert_to_tensor(preds)
        msk = decode_labels(preds, num_classes=args.num_classes)
        im = Image.fromarray(msk[0])
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        im.save(args.save_dir + str(count) + '.png')
        print(time.time() - a)
        print('The output file has been saved to {}'.format(args.save_dir + str(count) + 'mask.png'))
        count += 1
    
if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES']='3'
    main()
