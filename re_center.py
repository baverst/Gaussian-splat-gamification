import numpy
from plyfile import PlyData, PlyElement 

def recenter_axis(ply_content, axis, offset):
    axis_poses = ply_content.elements[0].data[axis]
    axis_avg = axis_poses.mean()
    axis_poses -= axis_avg
    axis_poses += + offset

def main():
    path_name = "UnityGaussianSplatting-main/UnityGaussianSplatting-main/projects/GaussianExample/Assets/CustomAssets/Truck/"
    filename = path_name +"left_wheel_centered"
    original_file_path =  filename +'.ply'
    copy_file_path = filename + '_and_unlit04.ply'
    
    offset_x = 0.023
    offset_y = -0.027
    offset_z = -0.011

    ply_content = PlyData.read(original_file_path)
    for i in range(0,44):
        ply_content.elements[0].data["f_rest_"+str(i)] = 0
    recenter_axis(ply_content, 'x', offset_x)
    recenter_axis(ply_content, 'y', offset_y)
    recenter_axis(ply_content, 'z', offset_z)

    PlyData.write(ply_content, copy_file_path)
    return
if __name__ == '__main__':
    main()