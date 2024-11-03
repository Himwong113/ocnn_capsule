import open3d as o3d
import numpy as np
import torch
import math
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
        print(int(math.pow(8,(self.max_depth-1))))
        return self.tensor_data[:,:int(math.pow(8,(self.max_depth-1)))]


N = 2048

#armadillo = o3d.data.ArmadilloMesh()
#mesh = o3d.io.read_triangle_mesh(armadillo.path)
#_pcd = mesh.sample_points_poisson_disk(N)

# Usage example:
_pcd = torch.rand(N,3)

processor = PointCloudProcessor(point_cloud=_pcd, num_points=N, max_depth=4)
tensor_data = processor.get_tensor_data()
print(processor.index_vec)
print(tensor_data.shape)

def create_zero_tensor(n):
    # Create a list of sizes with 'n' occurrences of 8
    size = [8] * n
    # Create and return the zero tensor of the specified size
    return torch.zeros(*size)

zerotens= create_zero_tensor(4)
print(zerotens.shape)
zerotens=zerotens.flatten()
print(zerotens.shape)

import math
for i in range (int(math.pow(8,4))):
    temp = create_zero_tensor(4).flatten()
    temp [i]=1


    matches = torch.all(tensor_data == temp, dim=1)
    matching_indices = torch.where(matches)[0]
    print(f'{i}:{matching_indices}')




