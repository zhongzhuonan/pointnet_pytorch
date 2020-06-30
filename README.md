# pointnet_pytorch
自行实现的pointnet，数据集为modelnet40。

参考：
[charlesq34.tensorflow](https://github.com/charlesq34/pointnet)
[fxia22.pytorch](https://github.com/fxia22/pointnet.pytorch)

数据modelnet40，讲off格式转换为ply格式。
目前随机采样2048个点，batchsize=64，训练400个epoch。准确率能达到77%左右。

1. 目前未使用数据增强；
2. 对调参也不是太熟悉；
3. 数据集转换为ply格式，可能是转换的方法不太对，导致点比较稀疏。
