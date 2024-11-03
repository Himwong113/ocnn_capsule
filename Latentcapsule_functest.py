import open3d as o3d
import numpy as np





import torch

def create_zero_tensor(n):
    # Create a list of sizes with 'n' occurrences of 8
    size = [8] * n
    # Create and return the zero tensor of the specified size
    return torch.zeros(*size)





def f_traverse(node, node_info):

    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            """
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            print(f'node.indices[{node_info.depth}] ={node.indices}')
            """



            for index in node.indices:
                index_vec[index].append(node_info.child_index)


            # we only want to process nodes / spatial regions with enough points
            #early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            """
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
            #print(f'elif node.indices[{node_info.depth}] ={node.indices}')
            """
        for index in node.indices:
            index_vec[index].append(node_info.child_index)
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop
def pointcloudlocation(pcd,num_point,depth=4):


    octree = o3d.geometry.Octree(max_depth=depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)



    octree.traverse(f_traverse)
    return index_vec



print('input')
N = 2000

armadillo = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo.path)
_pcd = mesh.sample_points_poisson_disk(N)
_pcd.scale(1 / np.max(_pcd.get_max_bound() - _pcd.get_min_bound()),
          center=_pcd.get_center())

depth =4


index_vec = [list() for _ in range(N)]
index_vec = pointcloudlocation(pcd=_pcd,num_point=N,depth=depth)

print(index_vec)
for i in range(len(index_vec)):
    temp =create_zero_tensor(depth)
    temp[tuple(index_vec[i])] =1
    temp=temp.flatten()
    index_vec[i]=temp


index_vec =torch.stack(index_vec)

print(index_vec)




"""
for i in range(N):
    print( f'{i} point : {octree.locate_leaf_node(pcd.points[i])}')
"""
