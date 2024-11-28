import numpy
from plyfile import PlyData, PlyElement 

def recenter_axis(ply_content, axis):
    axis_poses = ply_content.elements[0].data[axis]
    axis_avg = axis_poses.mean()
    axis_poses -= axis_avg

def main():
    path_name = "UnityGaussianSplatting-main/UnityGaussianSplatting-main/projects/GaussianExample/Assets/CustomAssets/"
    original_file_path = path_name + 'kitchen-point_cloud-iteration_7000-point_cloud-base.ply'
    copy_file_path = path_name + 'base_centered.ply'
    ply_content = PlyData.read(original_file_path)
    
    recenter_axis(ply_content, 'x')
    recenter_axis(ply_content, 'y')
    recenter_axis(ply_content, 'z')

    PlyData.write(ply_content, copy_file_path)
    return
if __name__ == '__main__':
    main()