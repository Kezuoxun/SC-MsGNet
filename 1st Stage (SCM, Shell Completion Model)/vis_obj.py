import open3d as o3d
import numpy as np
import os
from graspnetAPI import GraspNet
from PIL import Image
import pptk

def visualize_single_object(dataset_root, object_name):
    # 初始化 GraspNet 數據集
    g = GraspNet(root=dataset_root, camera='realsense', split='train')
    color = np.array(Image.open('/media/ntfs/graspnet/models/003/texture_map.png'))

    # 獲取物體模型路徑
    model_dir = os.path.join(dataset_root, 'models')
    object_path = os.path.join(model_dir, object_name, 'nontextured.ply')
    
    if not os.path.exists(object_path):
        print(f"物體模型 {object_name} 不存在.")
        return
    
    # 讀取點雲
    pcd = o3d.io.read_point_cloud(object_path)
    points = np.asarray(pcd.points)
    # 檢查點雲是否已有顏色
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        # 如果沒有顏色，添加隨機顏色
        colors = np.random.uniform(0, 1, size=(len(points), 3))
    
    # 使用pptk顯示點雲
    v = pptk.viewer(points, colors)
    # 設置一些視圖參數 (可選)
    v.set(bg_color=[0.1, 0.1, 0.1, 1])  # 設置背景顏色為深灰色
    # v.set(show_axis=False)  # 關閉 x、y、z 軸的顯示

if __name__ == "__main__":
    # 設置 GraspNet 數據集的根目錄
    dataset_root = "/media/ntfs/graspnet"
    
    # 指定要可視化的物體名稱
    object_name = "003"  # 請替換為您想要可視化的物體名稱
    
    visualize_single_object(dataset_root, object_name)