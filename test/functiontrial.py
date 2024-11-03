import ocnn
import torch
from utils import get_batch_octree

octree=get_batch_octree()
print(octree.keys[6].shape)