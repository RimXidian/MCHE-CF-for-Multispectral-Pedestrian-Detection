# MCHE-CF for Multispectral Pedestrian Detection

## Introduction

Multiscale Cross-modal Homogeneity Enhancement and Confidence-aware Fusion for Multispectral Pedestrian Detection
- paper download: https://ieeexplore.ieee.org/abstract/document/10114594


# Usage
## 1. Dependencies
This code is tested on [Ubuntu18.04 LTS，MATLAB R2018b，python 3.7，pytorch 1.5，CUDA 10.1]. 
 
 
 >make sure the GPU enviroment is the same as above, otherwise you may have to compile the `nms` and `utils` according to https://github.com/ruotianluo/pytorch-faster-rcnn. 
 ```
1. conda activate [your_enviroment]
2. pip install -r requirments.txt
```

## 2. Prerequisites
You need to prepare the dataset with the instructions in [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) to prepare KAIST dataset. 

## 3. Pretrained Model
We use VGG16 pretrained models in our experiments. You can download the model from:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)
* pretrained: [pretrained model](https://pan.baidu.com/s/169SszWgskGowMKIODRTppw), (extract code: `aaaa`)

Download them and put them into the data/pretrained_model/.

## 4. Train
Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```
Train the dataset using following commands:
```
python trainval_net.py
```
You can download the best model for the KAIST dataset in the paper [here](https://pan.baidu.com/s/169SszWgskGowMKIODRTppw), (extract code: `aaaa`)
## 5. Test
Test the dataset using following commands:
```
python gen_result.py
```
The result will generate at './result', then use the matlab [code](https://github.com/CalayZhou/MBNet/tree/master/KAISTdevkit-matlab-wrapper) to test the model.


# Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{MCHE-CF for Multispectral Pedestrian Detection,
    author = {Ruimin Li, Jiajun Xiang, Feixiang Sun, Ye Yuan, Longwu Yuan, Shuiping Gou},
    title = {Multiscale Cross-modal Homogeneity Enhancement and Confidence-aware Fusion for Multispectral Pedestrian Detection},
    booktitle = IEEE Transactions on Multimedia,
    year = {2023}
}
```

