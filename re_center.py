import numpy
from plyfile import PlyData, PlyElement 

def recenter_axis_average(ply_content, axis):
    axis_poses = ply_content.elements[0].data[axis]
    axis_avg = axis_poses.mean()
    axis_poses -= axis_avg

def recenter_axis_box(ply_content, axis):
    axis_poses = ply_content.elements[0].data[axis]
    axis_middle = (axis_poses.max()+axis_poses.min())/2
    axis_poses -= axis_middle

def recenter_axis_offset(ply_content, axis, offset):
    axis_poses = ply_content.elements[0].data[axis]
    axis_poses += offset


def main():
    path_name = "../tmp/UnityGaussianSplatting/projects/GaussianExample/Assets/for_paper/"
    filename = path_name +"left_wheel"
    original_file_path =  filename +'.ply'
    copy_file_path = filename + '_centered_offset.ply'
    useBox = False
    offset_x =0.445
    offset_y =-1.097
    offset_z =-1.944

    ply_content = PlyData.read(original_file_path)
    for i in range(0,44):
        ply_content.elements[0].data["f_rest_"+str(i)] = 0
    recenter_axis = recenter_axis_box if useBox else recenter_axis_average
    recenter_axis_offset(ply_content, 'x', offset_x)
    recenter_axis_offset(ply_content, 'y', offset_y)
    recenter_axis_offset(ply_content, 'z', offset_z)

    PlyData.write(ply_content, copy_file_path)
    return
if __name__ == '__main__':
    main()