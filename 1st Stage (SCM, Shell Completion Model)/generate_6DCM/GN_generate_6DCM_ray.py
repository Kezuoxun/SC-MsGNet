import os
import pickle
import numpy as np
import open3d as o3d
from PIL import Image
from PIL import ImageFile
from scipy import interpolate
import matplotlib.pyplot as plt
from graspnetAPI import GraspNet
from utils.utils import transform_points, parse_posevector
from utils.xmlhandler import xmlReader
from utils.time_controler import timeout
import pptk

import time
import queue
import multiprocessing as mp

debug = 1

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path, frameId):
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
    with open(os.path.join(pkl_path, '%04d.pkl'%frameId), 'wb') as f:
        pickle.dump(data, f)


def save_comap(data, pkl_path, frameId):
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
    if not os.path.exists(os.path.join(pkl_path, 'cmpf')):
        os.makedirs(os.path.join(pkl_path, 'cmpf'))
    if not os.path.exists(os.path.join(pkl_path, 'cmpb')):
        os.makedirs(os.path.join(pkl_path, 'cmpb'))

    cmpf = Image.fromarray((data[..., :3]*255).astype(np.uint8))
    cmpb = Image.fromarray((data[..., 3:]*255).astype(np.uint8))
    cmpf.save(os.path.join(pkl_path, 'cmpf', '%04d.png' % frameId))
    cmpb.save(os.path.join(pkl_path, 'cmpb', '%04d.png' % frameId))


def read_rgb_np(rgb_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(rgb_path).convert('RGB')
    img = np.array(img, np.uint8)
    return img


def read_mask_np(mask_path):
    mask = Image.open(mask_path)
    mask_seg = np.array(mask).astype(np.int32)
    #np.savetxt(r'mask_seg.txt',mask_seg, fmt='%f')
    return mask_seg


def read_pose(rot_path, tra_path):
    rot = np.loadtxt(rot_path, skiprows=1)
    tra = np.loadtxt(tra_path, skiprows=1) / 100.
    return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)


# @timeout(60*5)
def generate_coordinate_maps_gn(gn, sceneId, frameId, camera, k, pmap):
    print('scene = %03d, frame = %03d'%(sceneId, frameId))
    rgb_pth = os.path.join(gn.root, 'scenes', 'scene_' + str(sceneId).zfill(4), camera, 'rgb', '%04d.png'%frameId)
    rgb = read_rgb_np(rgb_pth)
    
    dpt_pth = os.path.join(gn.root, 'scenes', 'scene_' + str(sceneId).zfill(4), camera, 'label', '%04d.png'%frameId)
    mask_seg = read_mask_np(dpt_pth)
    #np.savetxt(r'mask.txt',mask_seg, fmt='%d')
    
    scene_reader = xmlReader(os.path.join(gn.root, 'scenes', 'scene_%04d' % sceneId, camera, 'annotations', '%04d.xml'% frameId))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    pose_list = []
    model_list = []
    for posevector in posevectors:
        obj_id, pose = parse_posevector(posevector)
        obj_list.append(obj_id)
        pose_list.append(pose)
    model_list = gn.loadObjModels(obj_list, simplify=True)
    
    #vis=o3d.visualization.Visualizer()
    #vis.create_window(width=1280, height=720)
    #vis.add_geometry(model_list[0])
    #o3d.visualization.draw_geometries([*model_list])

    comap_scene = None

    if debug:
        print('obj_id: ', obj_list)
    for i in range(len(obj_list)):
        if debug:
            print('obj: ',obj_list[i])
        if obj_list[i] == 46:
            model_list[i] = model_list[i].voxel_down_sample(voxel_size=0.001)
        if obj_list[i] == 0 or obj_list[i] == 40:
            # continue
            # distances = model_list[i].compute_nearest_neighbor_distance()
            avg_dist = np.mean(model_list[i].compute_nearest_neighbor_distance())
            radius = 3 * avg_dist
            bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(model_list[i], o3d.utility.DoubleVector(
                [radius, radius * 2]))
            bpa_mesh.remove_degenerate_triangles()
            bpa_mesh.remove_duplicated_triangles()
            bpa_mesh.remove_duplicated_vertices()
            bpa_mesh.remove_non_manifold_edges()
            #model_list[i] = bpa_mesh.sample_points_uniformly(number_of_points=int(len(model_list[i].points) * 1.3))
            if obj_list==0:
                model_list[i] = bpa_mesh.sample_points_uniformly(number_of_points=500000)
            else:
                model_list[i] = bpa_mesh.sample_points_uniformly(number_of_points=250000)


        mask = np.asarray(mask_seg == obj_list[i]+1, np.float32)
        if((mask==0).all()):     # If obj doesn't in the image, jump to next iteration.
            continue
        mask_vis = mask.copy()
        mask_vis = mask_vis.reshape(mask_vis.shape+(1,))
        #np.savetxt(r'mask.txt',mask, fmt='%f')
        pose = np.asarray(pose_list[i][:3])

        original = np.asarray(model_list[i].points).T
        original_nor = (original - original.min(1)[:, None]) / (original.max(1)[:, None] - original.min(1)[:, None])
        original_tran = np.concatenate((original[:3, :], np.ones((1, original.shape[1]))), axis=0)

        original_tran = pose @ original_tran
        project_mask = k @ original_tran
        project_mask = np.array(project_mask[:2] / project_mask[-1]).astype(int)

        unique, indices, counts = np.unique(project_mask, axis=1, return_inverse=True, return_counts=True)
        cmp = np.zeros_like(rgb).astype(np.float32)
        cow, coh = np.meshgrid(np.arange(0, 1280-1), np.arange(0, 720-1))

        grid = np.stack((cow, coh), -1)
        kernel = np.array([[-1, -1, -1, 0, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        slider = grid[..., None] + kernel[None, None]
        sort = np.argsort(indices)
        cum = np.append(0, counts).cumsum()
        arrayidx = cum[:-1, None] + np.arange(counts.max())[None]
        arrayidx = np.where(arrayidx >= cum[1:, None], cum[-1], arrayidx)

        dist = np.sqrt((original_tran ** 2).sum(0))
        localbest = np.append(dist[sort], -1)[arrayidx].argmax(1)
        globalbest = arrayidx[np.arange(cum.shape[0] - 1), localbest]
        best2D = project_mask[:, sort][:, globalbest]
        best3D = original_nor[:, sort][:3, globalbest]
        best3D = best3D[:, (best2D[1] < 720) * (best2D[0] < 1280)]  # Filter the out of bound coordinates
        best2D = best2D[:, (best2D[1] < 720) * (best2D[0] < 1280)]  # Filter the out of bound coordinates
        best3D = best3D[:, (-720 <= best2D[1]) * (-128 <= best2D[0])]  # Filter the out of bound coordinates
        best2D = best2D[:, (-720 <= best2D[1]) * (-128 <= best2D[0])]  # Filter the out of bound coordinates
        best3D = best3D.T * 255
        cmpb = cmp.copy()
        cmpM = cmp.copy()
        cmpb[best2D[1], best2D[0]] = best3D
        cmp9 = cmpb[slider[:, :, 1], slider[:, :, 0]]
        median = np.argsort(np.linalg.norm(cmp9, axis=3))[..., 4]
        cmpM[coh, cow] = cmp9[grid[..., 1] - 1, grid[..., 0] - 1, median]
        std_map = np.std(cmpb - cmpM, -1)
        idx_map = std_map > std_map[std_map > 0].mean()
        # idx_map = std_map > std_map[std_map > std_map[std_map > std_map[std_map > 0].mean()].mean()].mean()
        cmpb[idx_map] = cmpM[idx_map]

        dist = 1 / (dist + 1e-9)
        localbest = np.append(dist[sort], -1)[arrayidx].argmax(1)
        globalbest = arrayidx[np.arange(cum.shape[0] - 1), localbest]
        best2D = project_mask[:, sort][:, globalbest]
        best3D = original_nor[:, sort][:3, globalbest]
        best3D = best3D[:, (best2D[1] < 720) * (best2D[0] < 1280)]  # Filter the out of bound coordinates
        best2D = best2D[:, (best2D[1] < 720) * (best2D[0] < 1280)]  # Filter the out of bound coordinates
        best3D = best3D[:, (-720 <= best2D[1]) * (-128 <= best2D[0])]  # Filter the out of bound coordinates
        best2D = best2D[:, (-720 <= best2D[1]) * (-128 <= best2D[0])]  # Filter the out of bound coordinates
        best3D = best3D.T * 255
        cmpf = cmp.copy()
        cmpM = cmp.copy()
        cmpf[best2D[1], best2D[0]] = best3D
        cmp9 = cmpf[slider[:, :, 1], slider[:, :, 0]]
        median = np.argsort(np.linalg.norm(cmp9, axis=3))[..., 4]
        cmpM[coh, cow] = cmp9[grid[..., 1] - 1, grid[..., 0] - 1, median]
        std_map = np.std(cmpf - cmpM, -1)
        idx_map = std_map > std_map[std_map > 0].mean()
        # idx_map = std_map > std_map[std_map > std_map[std_map > std_map[std_map > 0].mean()].mean()].mean()
        cmpf[idx_map] = cmpM[idx_map]

        hs, ws = np.nonzero(mask)
        hmin, hmax = np.min(hs), np.max(hs)
        wmin, wmax = np.min(ws), np.max(ws)

        pmask = (pmap * mask[..., None])[hmin:hmax, wmin:wmax]
        cmp = (np.concatenate([cmpf, cmpb], -1) * mask[..., None])[hmin:hmax, wmin:wmax]
        begin = [hmin, wmin]

        mask = np.zeros([rgb.shape[0], rgb.shape[1], pmask.shape[-1]], np.float32)
        comap = np.zeros([rgb.shape[0], rgb.shape[1], cmp.shape[-1]], np.float32)
        if comap_scene is None:
            comap_scene = np.zeros([rgb.shape[0], rgb.shape[1], cmp.shape[-1]], np.float32)

        # mask[hmin:hmax, wmin:wmax] = pmask
        comap[hmin:hmax, wmin:wmax] = cmp/255
        comap = comap * mask_vis
        comap_scene += comap
        # print('ok')

    print('scene %04d, frame %04d ok' % (sceneId, frameId))
    save_comap(comap_scene, os.path.join(gn.root, 'comap', 'scene_'+str(sceneId).zfill(4), camera), frameId)
    return rgb, mask, comap_scene, begin, pose


def sample_render_tudl(rgb_pth, dpt_pth, RT_path, pcd_3d, pmap, k, idx=None):
    # rgb_pth = os.path.join(render_dir, image_db[idx]['rgb_pth'])
    # dpt_pth = os.path.join(render_dir, image_db[idx]['msk_pth'])
    rgb = read_rgb_np(rgb_pth).astype(np.float)
    mask_seg = np.asarray(read_mask_np(dpt_pth) > 0, np.int32)
    mask = np.asarray(mask_seg[..., 1] == 255, np.int32)
    # K = image_db[idx]['K']
    # pose = image_db[idx]['RT']
    pose = np.array([[9.99584750e-01, 2.52761095e-02, -1.38364016e-02, -1.05968000e+01],
                                      [ 2.29813551e-02, -9.88962181e-01, -1.46375073e-01,  4.21048000e+00],
                                      [-1.73834703e-02,  1.45996311e-01, -9.89132394e-01,  1.84081000e+01]])

    pc_3d_nor = (pcd_3d - pcd_3d.min(1)[:, None]) / (pcd_3d.max(1)[:, None] - pcd_3d.min(1)[:, None])
    # original_nor = (original - original.min(1)[:, None]) / (original.max(1)[:, None] - original.min(1)[:, None]) * (1 - (-1)) + (-1)
    pc_3d_tran = np.concatenate((pcd_3d[:3, :], np.ones((1, pcd_3d.shape[1]))), axis=0)

    pc_3d_tran = pose @ pc_3d_tran
    project_mask = k @ pc_3d_tran
    project_mask = np.array(project_mask[:2] / project_mask[-1]).astype(int)

    unique, indices, counts = np.unique(project_mask, axis=1, return_inverse=True, return_counts=True)
    cmp = np.zeros_like(rgb).astype(np.float)
    cow, coh = np.meshgrid(np.arange(0, 3383), np.arange(0, 2709))

    grid = np.stack((cow, coh), -1)
    kernel = np.array([[-1, -1, -1, 0, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
    slider = grid[..., None] + kernel[None, None]
    sort = np.argsort(indices)
    cum = np.append(0, counts).cumsum()
    arrayidx = cum[:-1, None] + np.arange(counts.max())[None]
    arrayidx = np.where(arrayidx >= cum[1:, None], cum[-1], arrayidx)

    dist = np.sqrt((pc_3d_tran ** 2).sum(0))
    localbest = np.append(dist[sort], -1)[arrayidx].argmax(1)
    globalbest = arrayidx[np.arange(cum.shape[0] - 1), localbest]
    best2D = project_mask[:, sort][:, globalbest]
    best3D = pc_3d_nor[:, sort][:3, globalbest].T * 255
    cmpb = cmp.copy()
    cmpM = cmp.copy()
    cmpb[best2D[1], best2D[0]] = best3D
    cmp9 = cmpb[slider[:, :, 1], slider[:, :, 0]]
    median = np.argsort(np.linalg.norm(cmp9, axis=3))[..., 4]
    cmpM[coh, cow] = cmp9[grid[..., 1] - 1, grid[..., 0] - 1, median]
    std_map = np.std(cmpb - cmpM, -1)
    idx_map = std_map > std_map[std_map > 0].mean()
    # idx_map = std_map > std_map[std_map > std_map[std_map > std_map[std_map > 0].mean()].mean()].mean()
    cmpb[idx_map] = cmpM[idx_map]

    dist = 1 / (dist + 1e-9)
    localbest = np.append(dist[sort], -1)[arrayidx].argmax(1)
    globalbest = arrayidx[np.arange(cum.shape[0] - 1), localbest]
    best2D = project_mask[:, sort][:, globalbest]
    best3D = pc_3d_nor[:, sort][:3, globalbest].T * 255
    cmpf = cmp.copy()
    cmpM = cmp.copy()
    cmpf[best2D[1], best2D[0]] = best3D
    cmp9 = cmpf[slider[:, :, 1], slider[:, :, 0]]
    median = np.argsort(np.linalg.norm(cmp9, axis=3))[..., 4]
    cmpM[coh, cow] = cmp9[grid[..., 1] - 1, grid[..., 0] - 1, median]
    std_map = np.std(cmpf - cmpM, -1)
    idx_map = std_map > std_map[std_map > 0].mean()
    # idx_map = std_map > std_map[std_map > std_map[std_map > std_map[std_map > 0].mean()].mean()].mean()
    cmpf[idx_map] = cmpM[idx_map]

    hs, ws = np.nonzero(mask)
    hmin, hmax = np.min(hs), np.max(hs)
    wmin, wmax = np.min(ws), np.max(ws)

    rgb = (rgb * mask[..., None])[hmin:hmax, wmin:wmax]
    pmask = (pmap * mask[..., None])[hmin:hmax, wmin:wmax]
    cmp = (np.concatenate([cmpf, cmpb], -1) * mask[..., None])[hmin:hmax, wmin:wmax]
    begin = [hmin, wmin]

    return rgb, pmask, cmp, begin, pose


def generate_coordinate_maps_gn_mp(gn, sceneId, camera, k, pmap, obj_ids):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'Scene {sceneId} Start !!!!!!!!')
    global debug
    if 0 in obj_ids:
        # return 0
        debug=1
        frame_queue = mp.Queue()
        for frameId in range(0, 256):
            if not os.path.exists(os.path.join(gn.root, 'comap', 'scene_'+str(sceneId).zfill(4), camera, 'cmpf', str(frameId).zfill(4)+'.png')):
                # generate_coordinate_maps_gn(gn, sceneId, frameId, camera, camK, pmap_g)
                frame_queue.put([gn, sceneId, frameId, camera, k, pmap])
            else:
                continue

        while not frame_queue.empty():
            batch_worker(frame_queue, obj_ids, worker_num=2)
    else:
        # return 0
        debug=0
        frame_queue = mp.Queue()
        for frameId in range(0,256):
            if not os.path.exists(os.path.join(gn.root, 'comap', 'scene_'+str(sceneId).zfill(4), camera, 'cmpf', str(frameId).zfill(4)+'.png')):
                frame_queue.put([gn, sceneId, frameId, camera, k, pmap])
            else:
                continue

        if not frame_queue.empty():
            batch_worker(frame_queue, obj_ids, worker_num=4, time_out=60*60*3.5)
    print(f'Scene {sceneId} Done !!!!!!')
    return 0


def batch_worker(my_q, obj_ids, worker_num=4, time_out=None):
    my_worker = list(range(worker_num))
    for worker in range(worker_num):
        my_worker[worker] = Worker(my_q, worker)
        my_worker[worker].start()
        if worker_num>1:
            time.sleep(30)

    for worker in range(worker_num):
        my_worker[worker].join(timeout=time_out)


class Worker(mp.Process):
    def __init__(self, queue, workerId):
        mp.Process.__init__(self)
        self.queue = queue
        self.workerId = workerId

    def run(self):
        while self.queue.qsize() > 0:
            # 取得新的資料
            try:
                msg = self.queue.get()
                generate_coordinate_maps_gn(gn=msg[0], sceneId=msg[1], frameId=msg[2], camera=msg[3], k=msg[4], pmap=msg[5])
            except:
                print('Something wrong!!')
            finally:
                time.sleep(1)


if __name__ == '__main__':

    # graspnet_root = '/media/user/2tb/graspnet'
    graspnet_root = '/media/dsp/JTsai/graspnet'
    camera = 'realsense'
    try:
        gn = GraspNet(graspnet_root, camera=camera, split='all')
    except:
        graspnet_root = '/media/dsp/JTsai/graspnet'
        gn = GraspNet(graspnet_root, camera=camera, split='all')
    camK = np.load(os.path.join(graspnet_root, 'scenes', 'scene_' + str(0).zfill(4), camera, 'camK.npy'))
    cow_g, coh_g = np.meshgrid(np.arange(0, 1280), np.arange(0, 720)) # 1280, 720 for GraspNet
    pmap_g = np.stack((cow_g / 1280., coh_g / 720., np.ones_like(cow_g)), -1)
    generate_coordinate_maps_gn(gn, 1, 9, camera, camK, pmap_g)

    for scene_id in range(100, 130):
        camK = np.load(os.path.join(graspnet_root, 'scenes', 'scene_' + str(scene_id).zfill(4), camera, 'camK.npy'))
        cow_g, coh_g = np.meshgrid(np.arange(0, 1280), np.arange(0, 720)) # 1280, 720 for GraspNet
        pmap_g = np.stack((cow_g / 1280., coh_g / 720., np.ones_like(cow_g)), -1)

        generate_coordinate_maps_gn_mp(gn, scene_id, camera, camK, pmap_g, gn.getObjIds(scene_id))


    
#   fig, axs = plt.subplots(2, 2)
#   axs[0, 0].imshow(rgb)
#   axs[0, 0].set_title('RGB image')
#   axs[0, 1].imshow(mask)
#   axs[0, 1].set_title('Object mask')
#   axs[1, 0].imshow(comap[..., :3])
#   axs[1, 0].set_title('F3DCM')
#   axs[1, 1].imshow(comap[..., 3:])
#   axs[1, 1].set_title('R3DCM')
    #fig.show()
    #fig.savefig('/home/dsp/Meng/generate_6DCM/s{}_f{}.png'.format(scene_id, frame_id))
    print('Ok')
