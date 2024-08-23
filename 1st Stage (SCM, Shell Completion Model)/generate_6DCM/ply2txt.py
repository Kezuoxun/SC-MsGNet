import os
import numpy as np
from graspnetAPI import GraspNet as GN
import open3d as o3d 

camera = 'kinect'
scene_id = 0
graspnet_root = '/media/dsp/B2B4107BB41043EF/graspnet'

g=GN(graspnet_root, camera=camera, split='all')
obj_ids = g.getObjIds(scene_id)
models = g.loadObjModels(obj_ids)
camK = np.load(os.path.join(graspnet_root, 'scenes', 'scene_' + str(scene_id).zfill(4), camera, 'camK.npy'))
print(camK)
out_arr = []
#for i in range(len(obj_ids)):
#    out_arr.append(np.asarray(models[i].points).T)
    #print(out_arr[i].shape)

#print(out_arr)
#for i in range(3):
#    np.savetxt(r'test%d.txt'%i,out_arr[i],fmt='%f')
print('Done')
