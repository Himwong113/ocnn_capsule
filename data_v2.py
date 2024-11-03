#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""

import os
import sys
import glob
import h5py
import numpy as np
import ocnn.octree
from torch.utils.data import Dataset
import json
import cv2
from ocnn.octree import Octree
from ocnn.octree import Points
import torch
import math
from visual import plotoctree
import open3d as o3d


#### classtification modelnet 40 part###
def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        print('dl')
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('curl --insecure -O %s' % www)
        os.system('unzip %s' % zipfile)
        os.system('curl %s; unzip %s' % (www, zipfile))
        # os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        # os.system('rm %s' % (zipfile))


def load_data(partition):
    # print('dl ()')
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        # print(all_data)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def key2tensor(point, depth):
    pointcloud = Points(torch.from_numpy(point))
    octree = Octree(depth=depth)
    octree.build_octree(point_cloud=pointcloud)
    keystring = [octree.key(i, True) for i in range(len(octree.keys))]
    return keystring


def octree_one_hot(point, depth):
    occup_idx = key2tensor(point, depth)  # [depth0 occup index, depth 2 occup index,....]
    zero = [torch.zeros(int(math.pow(math.pow(2, j), 3))) for j in
            range(depth + 1)]  # each leaf have (2l)^3 l=0,1,2,3,.... # 1,8,64,....
    for k in range(depth + 1):
        temp = zero[k]  # take depth k index
        temp[occup_idx[k]] = 1  # mark down 1 if occup for specfic depth index
        zero[k] = temp  # replace 0 tensor
    return torch.concat(zero)  # concat all index


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', treedepth: int = 4):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.depth = treedepth

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        # if self.partition == 'train':
        #    pointcloud = translate_pointcloud(pointcloud)
        #    np.random.shuffle(pointcloud)
        octree = octree_one_hot(pointcloud, self.depth).unsqueeze(dim=0)  # (B,index) -> (b,1,index)
        # print(np.max(pointcloud))
        # from visual import plotoctree
        # plotoctree(pointcloud,depth=7)

        return octree, label

    def __len__(self):
        return self.data.shape[0]


#### classtification modelnet 40 part END###

#### part segmetation Shapenet####
def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


def load_data_partseg(partition):
    download_shapenetpart()
    cnt = 0
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*val*.h5'))
    elif partition == 'test':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*test*.h5'))


    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*%s*.h5' % partition))

    cnt = 0
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')

        data = f['data'][:].astype('float32')

        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)

        # print(f'all data = {all_data}')
        all_label.append(label)
        all_seg.append(seg)

    # print(f'all data = {all_data}')
    all_data = np.concatenate(all_data, axis=0)
    print(f'all data ={all_data.shape}')
    all_label = np.concatenate(all_label, axis=0)
    print(f'all label ={all_label.shape}')
    all_seg = np.concatenate(all_seg, axis=0)
    print(f'all seg ={all_seg.shape}')
    return all_data, all_label, all_seg


def load_color_partseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/partseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    partseg_colors = np.array(colors)
    partseg_colors = partseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1350
    img = np.zeros((1350, 1890, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (1900, 1900), [255, 255, 255], thickness=-1)
    column_numbers = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    column_gaps = [320, 320, 300, 300, 285, 285]
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for row in range(0, img_size):
        column_index = 32
        for column in range(0, img_size):
            color = partseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.76, (0, 0, 0), 2)
            column_index = column_index + column_gaps[column]
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 50:
                cv2.imwrite("prepare_data/meta/partseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column + 1 >= column_numbers[row]):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break


class PointCloudProcessor:
    def __init__(self, point_cloud, num_points, max_depth=4):
        self.num_points = num_points
        self.max_depth = max_depth+1
        self.pcd = point_cloud
        self.index_vec = [list() for _ in range(num_points)]
        self.tensor_data = None
    """
    def load_point_cloud(self, path):
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_poisson_disk(self.num_points)
        pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
                  center=pcd.get_center())
        return pcd
    """


    def create_zero_tensor(self, n):
        size = [8] * n
        return torch.zeros(*size)

    def traverse_callback(self, node, node_info):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            for index in node.indices:
                self.index_vec[index].append(node_info.child_index)

        return False  # No early stopping

    def process_point_cloud(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pcd)
        octree = o3d.geometry.Octree(max_depth=self.max_depth)
        octree.convert_from_point_cloud(pcd, size_expand=0.01)
        octree.traverse(self.traverse_callback)

    def generate_tensors(self):
        for i in range(len(self.index_vec)):
            temp = self.create_zero_tensor(self.max_depth)
            if self.index_vec[i]:  # Ensure list is not empty
                temp[tuple(self.index_vec[i])] = 1
            temp = temp.flatten()
            self.index_vec[i] = temp
        self.tensor_data = torch.stack(self.index_vec)

    def get_tensor_data(self):
        if self.tensor_data is None:
            self.process_point_cloud()
            self.generate_tensors()

        return self.tensor_data[:,:int(math.pow(8,(self.max_depth-1)))]
class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='trainval', class_choice=None, depth=4):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.depth = depth
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice
        self.partseg_colors = load_color_partseg()

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        #print(pointcloud.shape)
        label = self.label[item]
        #print(f'seg ={self.seg.shape}')
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        octree = octree_one_hot(pointcloud, self.depth).unsqueeze(dim=0)  # (B,index) -> (b,1,index)
        #processor  = PointCloudProcessor(point_cloud=pointcloud,num_points=self.num_points,max_depth=self.depth)
        #pointcloudlocation = processor.get_tensor_data()

        return octree,pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


#### part segmetation Shapenet####
if __name__ == '__main__':

    # train = ModelNet40(8000)
    #
    # test = ModelNet40(1024, 'test')
    #
    ##from torch.utils.data import DataLoader
    #
    ##test_loader = DataLoader(ModelNet40(partition='train', num_points=8000), num_workers=8,
    #
    ##                         batch_size=32, shuffle=True, drop_last=False)
    #
    ##print(torch.max(test))
    #
    # for idx ,(data, label) in enumerate(test):
    #
    #    print(data.shape)
    #
    #    print(label.shape)
    #
    #    print(torch.max(data))

    import visual

    # print(data[0])

    # print(torch.nonzero(data))

    # if idx>=6:break

    trainval = ShapeNetPart(1024, 'trainval')
    # test = ShapeNetPart(1024, 'test')
    data, pcd,label, seg = trainval[0]
    print(data)
    # print(data.shape)
    # print(label.shape)
    # print(label)
    # print(seg.shape)
    # print(seg)
    label_one_hot = np.zeros((label.shape[0], 16))
    for idx in range(label.shape[0]):
        label_one_hot[idx, label[idx]] = 1
    print(seg)
    print(seg.shape)

    print(pcd)
    print(pcd.shape)

    print(label)
