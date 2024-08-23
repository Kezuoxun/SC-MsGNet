import os
import pptk
import numpy as np
from graspnetAPI import GraspNet as GN
import open3d as o3d 
from utils.utils import transform_points, parse_posevector
from utils.xmlhandler import xmlReader

x = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(x.shape)
y = np.asarray(x==0)
print(y.shape)
print(y)

camera = 'realsense'
scene_id = 19
annId = 0
graspnet_root = '/media/ntfs/graspnet'

g=GN(graspnet_root, camera=camera, split='all')
scene_model = g.loadSceneModel(scene_id, camera=camera, annId=annId, align=False)
obj_ids = g.getObjIds(scene_id)
obj_models = g.loadObjModels([70])

#print('model_shape: ', np.asarray(obj_models[1].points).T.shape)

scene_reader = xmlReader(os.path.join(graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera, 'annotations', '%04d.xml'% annId))
posevectors = scene_reader.getposevectorlist()
obj_list = []
pose_list = []
model_list = []
for posevector in posevectors:
    obj_id, pose = parse_posevector(posevector)
    obj_list.append(obj_id)
    pose_list.append(pose)

#print('pose_len: ', len(p    v = pptk.viewer(pcd, colors)cose_list))
print('pose 1: ',pose_list[1])
print('pose 1 array RT: ', np.asarray(pose_list[1][:3]))

camK = np.load(os.path.join(graspnet_root, 'scenes', 'scene_' + str(scene_id).zfill(4), camera, 'camK.npy'))
param = o3d.camera.PinholeCameraParameters()
param.extrinsic = np.eye(4,dtype=np.float64)
param.intrinsic.set_intrinsics(1280, 720, camK[0][0], camK[1][1], camK[0][2], camK[1][2])
#if camera == 'kinect':
#    param.intrinsic.set_intrinsics(1280,720,631.5,631.2,639.5,359.5)
#elif camera == 'realsense':
#    param.intrinsic.set_intrinsics(1280,720,927.17,927.37,639.5,359.5)

vis=o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720)
ctr = vis.get_view_control()

for i in range(len(obj_models)):
    #obj_models[i] = obj_models[i].voxel_down_sample(voxel_size=0.005)
    vis.add_geometry(obj_models[i])

for i in range(len(scene_model)):
    #scene_model[i] = scene_model[i].voxel_down_sample(voxel_size=0.005)
    vis.add_geometry(scene_model[i])

ctr.convert_from_pinhole_camera_parameters(param)
vis.poll_events()
filename = os.path.join('6D_test','{}_{}_6d.png'.format(0, camera))
vis.capture_screen_image(filename, do_render=True)
o3d.visualization.draw_geometries([*scene_model])
o3d.visualization.draw_geometries([*obj_models])
scene_pcd = scene_model[4]
scene_np = np.asarray(scene_pcd.points)

if scene_pcd.has_colors():
    colors_np = np.asarray(scene_pcd.colors)
else:
    # 如果點雲沒有顏色，可以生成隨機顏色或使用其他方法賦予顏色
    colors_np = np.random.rand(len(scene_np), 3)
# v=pptk.viewer(scene_np)
o3d.visualization.draw_geometries([scene_model[4]])
v = pptk.viewer(scene_np, colors_np)


print('Done')
