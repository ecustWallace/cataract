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

` python train.py `

If train from a pre-trained model:

` python train.py --model {your_model} `

### Inference

` python inference.py --output-dir {your_output_dir} --model {your_model} `
