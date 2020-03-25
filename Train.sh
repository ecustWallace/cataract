#! /bin/bash
cd CNN
cd tensorflow-deeplab-resnet-master
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deeplab
python inference_batch.py --model-weights /home/deeplab/CNN/tensorflow-deeplab-resnet-master/snapshots/train_8/model.ckpt-3000
