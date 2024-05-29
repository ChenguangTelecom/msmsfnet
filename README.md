## [Msmsfnet: a multi-stream and multi-scale fusion net for edge detection](https://arxiv.org/abs/2404.04856)

This is the PyTorch implementation of our proposed new network architecture for edge detection, msmsfnet, which achieves better performance in three publicly available datasets for edge detection than previous methods when all models are trained from scratch.

Should you have any questions, please do not hesitate to contact me: chenguang.liu.light@outlook.com .

## Requirements

PyTorch

opencv

tqdm


## Training & Testing

### Download the datasets

You can download the datasets for edge detection following the instructions [here](https://github.com/yun-liu/RCF). After downloading the datasets, put the training data into a folder like '/media/user/data/datasets/...', and put the testing images into the folder './data'. Then change the path to the training dataset folder in the bash file '*.sh' and change the name of the txt/lst file that contains the list of the training data accordingly.

### Training with single GPU

Run the following command to train the model using a single GPU:

bash run_msmsfnet.sh

### Training the model with multiple GPUs using DDP of PyTorch

Run the following command:

bash run_msmsfnet_ddp.sh

### Fine-tuning the model using pretrained weights

Run the following command:

bash run_msmsfnet_fine_tuning.sh

### Testing

Put the testing images into the './data' folder, and run the following command:

bash run_msmsfnet_test.sh
