# Contextual-Deeplab-V3

python=2.7

## Tutorial

### Download source code

``` git clone https://github.com/ecustWallace/cataract.git ```

### Download modules

``` pip install -r requirements.txt ```

### Download dataset and model checkpoint

#### Dataset

https://drive.google.com/file/d/13I2xBSyVNAL1YySyZgdHmfr5yl2s3z6G/view?usp=sharing

Make a directory called ` train ` in the root directory, and then extract the folder to ` train `.

#### Dataset Index 
https://drive.google.com/file/d/1ByE3X_MGa6gqRKFdH6QydXpKS6MNUHo_/view?usp=sharing

Put the file into ` train `.

#### Model Checkpoint 
https://drive.google.com/file/d/1LpNmBzcgZDrFef-5wnDFdWEVCJ4SeNXY/view?usp=sharing

Make a directory called ` snapshot ` in the root directory, and then extract this file into ` snapshot `.

### Train

If you want to train from scratch:

` python train.py`

If train from a pre-trained model:

` python train.py --restore-from {your_model} `

It can be directly changed in `train.py` by changing `RESTORE_FROM`.

### Inference

We can make the inference on any folder that contains the images. 

` python inference_batch.py --data-dir {your_data_dir} --save-dir {your_save_dir} --model {your_model_name} `

For example:

` python inference_batch.py --data-dir inference/output/train_12+_MP/N3_600_C/ --save-dir `
