# AM-MulFSNet: A Fast Semantic Segmentation Network Combining Attention Mechanism And Multi-branch


### Introduction
<p align="center"><img width="100%" src="./image/architecture.png" /></p>



### Installation
- Env: Python 3.8; PyTorch 1.10; CUDA 11.3; cuDNN V8
- Install some packages


### Dataset
You need to download the dataset——Cityscapes, and put the files in the `dataset` folder with following structure.
```
├── cityscapes
|    ├── gtCoarse
|    ├── gtFine
|    ├── leftImg8bit
|    ├── cityscapes_trainval_list.txt
|    ├── cityscapes_train_list.txt
|    ├── cityscapes_test_list.txt
|    └── cityscapes_val_list.txt           
```

