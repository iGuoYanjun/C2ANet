# C2ANet: Cross-scale and Cross-modality Aggregation Network for Scene Depth Super-Resolution
This repo implements the testing of depth upsampling networks for "C2ANet: Cross-scale and Cross-modality Aggregation Network for Scene Depth Super-Resolutionn" by Xinchen Ye and et al. at DLUT.
## Installation
The code requires pytorch>= 1.7.0 and python=3.7
The mmcv is request, which can be installed by 
`pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html`
Please replace the original deform_conv.py with the file we provided.
## quick test
`python test_middlebury.py`
## download pretrained models and test data
Link of weights: https://pan.baidu.com/s/1145CqN68BaCyVTO6rczgTg 
pwd：3cad
Link of middleury test data：https://pan.baidu.com/s/1u09_U3ljVFjxwOvnboYpEw 
pwd：u0r3
