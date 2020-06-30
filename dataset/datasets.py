'''
    # Time :    2020/6/5 上午9:47 
    # Author :  ZZN
    # File :    datasets.py
    # Version:  V0.0
    # Desc :
'''

import os
import torch
import torch.utils.data as data
import numpy as np
from plyfile import PlyData, PlyElement

'''
功能：1.从train.txt读取数据，得到一个从airplane到xbox的列表
     2.创建一个modelnet40_id.txt的文件，格式为airplane 0这种
'''
def get_modelnet_id(root):
    classes=[]
    with open(os.path.join(root,'train.txt'),'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes=np.unique(classes)
    with open('modelnet40_id.txt','w+') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i],i))

'''
***********************还需修改*************************
功能：数据的缩放归一化,[0,1之间].需要修改因为可能存在除以0的情况
输入：point_set :采样点x3
'''
def data_normalization(point_set):
    np.seterr(divide='ignore',invalid='ignore')
    point_min=np.min(point_set,axis=0) #axix=0,每列最小值
    point_max=np.max(point_set,axis=0)
    point_set_norm=(point_set-point_min)/(point_max-point_min)  #这样写可能遇到无效值 除以0这种情况
    return point_set_norm

'''
功能：ModelNet40数据集加载,继承在data.Dataset,需要重写len和getitem函数
'''
class ModelNet40(data.Dataset):
    def __init__(self,root,sample_points=1024,split='train'):
        self.root=root
        self.sample_point=sample_points
        self.split=split
        # 从train.txt或者test.txt获得数据
        self.datapath=[]
        with open(os.path.join(root,'{}.txt'.format(self.split)),'r') as f:
            for line in f:
                self.datapath.append(line.strip())

        # 打开model40_id.txt文件每行放入字典中,key-value这种形式
        self.cat={}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'modelnet40_id.txt'),'r') as f:
            for line in f:
                ls=line.strip().split()
                self.cat[ls[0]]=int(ls[1])
        # 字典到列表的转换
        self.classes=list(self.cat.keys())

    def __getitem__(self, idx):
        fn=self.datapath[idx] #当idx=0 如这种格式：glass_box/train/glass_box_0090.ply
        cls=self.cat[fn.split('/')[0]] #idx=0时 glass_box对应编号16。这里的[0]代表以‘/’为界限的第0个也就是类型名字
        with open(os.path.join(self.root,fn),'rb') as f:
            plydata=PlyData.read(f) # ply数据格式
        pts=np.vstack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).T  #ply数据，放成nx3的形式
        sample_points=np.random.choice(len(pts),self.sample_point,replace=True)  #第一个参数必须是一位数组
        point_set=pts[sample_points,:] #点集合变成：设置的采样点x3
        # point_set=data_normalization(point_set)
        point_set=torch.from_numpy(point_set.astype(np.float32))
        clc=torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set,clc

    def __len__(self):
        return len(self.datapath)


def main():
    root=r'/home/sirb/Documents/ModelNet40_ply'
    get_modelnet_id(root)
    train_data=ModelNet40(root)
    print(len(train_data))
    print(train_data[0])

if __name__=='__main__':
    main()