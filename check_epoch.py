import torch
import torchvision.models as models
import os
import torch.optim as optim
from models.graspnet import GraspNet_MSCG_1, GraspNet_MSCG_context

# 創建模型

CHECKPOINT_PATH  = '/home/dsp/6DCM_Grasp/zuo/MSCG/logs/log_ssd_AFF_MSCG_PT/minkuresunet_epoch17.tar'
# Load the Adam optimizer
net = GraspNet_MSCG_context(512, 15000,  num_view=300, cylinder_radius=0.08, hmin=-0.02, hmax=0.04,  is_training=True) 

optimizer = optim.Adam(net.parameters(), lr=0.001)

# 檢查是否存在checkpoint文件
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    # 載入checkpoint文件
    checkpoint = torch.load(CHECKPOINT_PATH)
    # 載入模型狀態字典和優化器狀態字典
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 獲取當前epoch數
    start_epoch = checkpoint['epoch']
    # 輸出模型的架構
    # print(net)
    # 打印模型的forward方法
    print(net.forward)
    print(start_epoch)
    # 可以進一步查看模型的每一層的參數
    # for name, param in net.named_parameters():
    #     print(name, param.size())