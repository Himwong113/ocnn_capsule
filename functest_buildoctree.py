import ocnn
from ocnn.octree import Octree
from ocnn.octree import Points
import torch
import math
from visual import plotoctree

depths =6
def key2tensor(point,depth):
    pointcloud = Points(point)
    octree =Octree(depth=depth
                   )
    octree.build_octree(point_cloud=pointcloud  )

    keystring =  [octree.key(i,True) for i in range(len(octree.keys))]

    #keystring = torch.concat(keystring)

    return keystring




def octree_one_hot(point,depth):
    occup_idx = key2tensor(point,depths)
    #print(occup_idx)

    zero = [torch.zeros(int(math.pow(math.pow(2,j),3))) for j in range(depth+1) ]


    for k in range(depth+1):
        temp=zero[k]
        temp[occup_idx[k]]=1
        zero[k]=temp


    return torch.concat(zero)

def octree_dec(point,depth):
    occup_idx = key2tensor(point,depths)
    #print(occup_idx)

    return occup_idx[-1]
#point =torch.rand(1024,3)
#occup =octree_one_hot(point,4)
#print(occup.shape)
#plotoctree(point)
#
#
#print([math.pow(math.pow(2,j),3) for j in range(4+1)])
#
#x = torch.tensor([1,2,3,4])
#y = torch.tensor([1,2,3,4])
#
#print(torch.where(x==y))
#print( torch.numel(torch.where(x==y)[0]))



def octree2point(arr):
    key = torch.tensor( [int("{0:b}".format(i)) for i in arr[0]])
    xyzb=ocnn.octree.key2xyz(key)
    xyz = list(xyzb[:3])
    xyz = torch.stack(xyz).permute((1,0))
    return xyz

"""
arr1 = torch.rand(1024,3)

depth=4
print(arr1.shape)
arr1 = arr1.permute((1,0))
print(arr1.shape)
key = ocnn.octree.xyz2key(x = arr1[0],y=arr1[1],z=arr1[2])
print(key.shape)
print( key)

"""


"""
key = torch.randint(0, 2**48, size=(5, 16), dtype=torch.long)
xyzb=ocnn.octree.key2xyz(key)

xyz = list(xyzb[:3])
print(xyz)

"""
#class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike',
#                 'mug', 'pistol', 'rocket', 'skateboard', 'table']
#seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
#index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
#
#import numpy as np
#choice = 'bag'
#
#print(seg_num[list(np.where(choice)[0])[0]])
##arr=octree_dec(arr1,depth=depth)
##
##
##print(arr)
###b,_,_ = arr.shape
##
##arr = torch.nonzero(arr,as_tuple=False).permute((1,0))
##
##
###print(arr)
##point=octree2point(arr)
##from visual import plot3d
##plot3d(arr1)
##plot3d(point)

import torch

def create_zero_tensor(num_point,depth):
    # Create a list of sizes with 'n' occurrences of 8
    dim = [8] * depth
    # Create and return the zero tensor of the specified size
    tree = torch.zeros(*dim).unsqueeze(dim=0)


    return tree


## Example usage
#tensor_4 = create_zero_tensor(4,1000)
##tensor_3 = create_zero_tensor(3,1000)
##
#print("Tensor when input is 4:", tensor_4.shape)
#print(f"Tensor when input is 4 :{tensor_4.flatten().unsqueeze(dim=-1).shape}")
#print("Tensor when input is 3:", tensor_3.shape)
#print(f"Tensor when input is 4 :{tensor_3.flatten().unsqueeze(dim=-1).shape}")

import numpy as np
#vec = [list() for _ in range(10)]  # Create a list of 10 empty lists
#vec[1].append(1)  # Append 1 to the list at index 1
#print(vec)
"""
x =torch.rand(12,1,4681).to('cuda')
x= x[:,:,-4096:]
print(f'x={x.shape}')
x= [1,2,3,4,5,6,7,8,9]
x=x[-3:]
print(f'x={x}')
for i in range(8,0,-2):
    print(i)
"""
