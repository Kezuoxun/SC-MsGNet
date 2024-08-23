import numpy as np
import pickle
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import threading
import queue
import time
import cv2
import skimage.exposure
import open3d as o3d 
import os
import pptk

# graspnet_root = '/media/dsp/B2B4107BB41043EF/graspnet'

test_pkl = 0
test_array = 0
test_depth = 1
pkl_checker = 0
comap_checker = 1
test_seg=1

if test_seg:
    
    # 假设 semantic_mask 是你的灰度语义掩码，值的范围在 [0, 2] 之间
    sm =  cv2.imread('/media/ntfs/graspnet/scenes/scene_0125/realsense/label/0000.png', cv2.IMREAD_UNCHANGED) # 你的语义掩码，形状为 (H, W)

    print(type(sm[0][0]))
    print(sm[0][0]*0.001)
    print(sm.shape)

    stretch = skimage.exposure.rescale_intensity(sm, in_range='image', out_range=(0,255)).astype(np.uint8)

# convert to 3 channels
    stretch = cv2.merge([stretch,stretch,stretch])

# define colors
    color1 = (0, 0, 255)     #red
    color2 = (0, 165, 255)   #orange
    color3 = (0, 255, 255)   #yellow
    color4 = (255, 255, 0)   #cyan
    color5 = (255, 0, 0)     #blue
    color6 = (128, 64, 64)   #violet
    colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)

# resize lut to 256 (or more) values
    lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)

# apply lut
    result = cv2.LUT(stretch, lut)

# create gradient image
    grad = np.linspace(0, 255, 512, dtype=np.uint8)
    grad = np.tile(grad, (20,1))
    grad = cv2.merge([grad,grad,grad])

# apply lut to gradient for viewing
    grad_colored = cv2.LUT(grad, lut)
    
    plt.imshow(sm)
    plt.colorbar(label='Distance to Camera')
    plt.show()


if test_pkl:
    with open('/media/ntfs/graspnet/comap/scene_0000/realsense/0000.pkl', 'rb') as f:
        a=pickle.load(f)
        print(a.shape)

        mask_path = '/media/ntfs/B2B4107BB41043EF/graspnet/scenes/scene_0000/realsense/label/0000.png'
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(np.int16)
        mask = np.asarray((mask == 1), np.int16)
        print(mask.shape)

        #a=a*np.expand_dims(mask, axis=2)

        #label = np.append(a, np.expand_dims(mask, axis=2), axis=2)
        #print(label[:,:,6].shape)
        
        xyz1 = np.asarray(a[...,:3]).reshape(-1,3)
        xyz2 = np.asarray(a[...,3:]).reshape(-1,3)
        #print(xyz1.shape)

        v= pptk.viewer(xyz1)
        v= pptk.viewer(xyz2)

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(xyz1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(xyz2)

        camK = np.load('/media/ntfs//graspnet/scenes/scene_0000/realsense/camK.npy')
        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = np.eye(4,dtype=np.float64)
        param.intrinsic.set_intrinsics(1280, 720, camK[0][0], camK[1][1], camK[0][2], camK[1][2])
        
        vis=o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=720)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()

        o3d.visualization.draw_geometries([pcd1, pcd2])

    
        plt.imshow(a[...,:3])
        #plt.show()
        plt.imshow(a[...,3:])
        #plt.show()


if test_array:
    a = np.array([[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]])
    print('a:', a)
    a = a*100
    print('a:', a)
    b = [0,1,2,3,4,5,6,7]
    print('b: ', b)
    del b[::3]
    print('b: ', b)


if test_depth:
    depth_path = '/media/ntfs/graspnet/scenes/scene_0125/realsense/depth/0000.png'
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    #depth = Image.open(depth_path)
    print(type(depth[0][0]))
    print(depth[0][0]*0.001)
    print(depth.shape)

    stretch = skimage.exposure.rescale_intensity(depth, in_range='image', out_range=(0,255)).astype(np.uint8)

# convert to 3 channels
    stretch = cv2.merge([stretch,stretch,stretch])

# define colors
    color1 = (0, 0, 255)     #red
    color2 = (0, 165, 255)   #orange
    color3 = (0, 255, 255)   #yellow
    color4 = (255, 255, 0)   #cyan
    color5 = (255, 0, 0)     #blue
    color6 = (128, 64, 64)   #violet
    colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)

# resize lut to 256 (or more) values
    lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)

# apply lut
    result = cv2.LUT(stretch, lut)

# create gradient image
    grad = np.linspace(0, 255, 512, dtype=np.uint8)
    grad = np.tile(grad, (20,1))
    grad = cv2.merge([grad,grad,grad])

# apply lut to gradient for viewing
    grad_colored = cv2.LUT(grad, lut)
    
    plt.imshow(depth)
    plt.colorbar(label='Distance to Camera')
    plt.show()


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

if pkl_checker:
    for s in range(0,56):
        for f in range(0,256):
            path = os.path.join('/media/ntfs/graspnet/comap', 'scene_%04d'%s, 'realsense', '%04d.pkl'%f)
            print('s: %04d, f: %04d'%(s,f))
            a = read_pickle(path)

def read_comap_np(rgb_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(rgb_path).convert('RGB')
    img = np.array(img, np.uint8)
    img = (img.astype(np.float32)/255)
    return img

if comap_checker:
    s = 125
    f = 0
    path = os.path.join('/media/ntfs/graspnet/comap', 'scene_%04d' % s, 'realsense')
    cmpf = read_comap_np(os.path.join(path, 'cmpf', '%04d.png' % f))
    cmpb = read_comap_np(os.path.join(path, 'cmpb', '%04d.png' % f))
    plt.imshow(cmpf)
    plt.show()
    plt.imshow(cmpb)
    plt.show()


