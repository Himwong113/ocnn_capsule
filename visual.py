import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

# Plot point cloud
def plot3d(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of points
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot title
    ax.set_title('Point Cloud in Octree Representation')

    plt.show()


def plotoctree(point_cloud,color=np.random.rand(1024, 3),depth=4):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Create an octree
    octree = o3d.geometry.Octree(max_depth=depth)
    pcd.colors = o3d.utility.Vector3dVector(color)

    # Convert the point cloud to an octree representation
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    # Visualize the octree
    o3d.visualization.draw_geometries([octree])


if __name__ == "__main__":
    import numpy as np


    # Generate random point cloud
    num_points = 1024
    point_cloud = np.random.rand(num_points, 3)  # Random points in 3D space
    #point_cloud = o3d.io.read_point_cloud("airplane_0001.ply")
    # Create Open3D point cloud
    plotoctree(point_cloud)

