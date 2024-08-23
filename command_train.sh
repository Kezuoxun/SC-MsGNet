
# CUDA_VISIBLE_DEVICES=0 python train.py --camera realsense --log_dir logs/log_GraspNet_MSCG_MSGE_no_scm --batch_size 4   --learning_rate 0.001  --model_name minkuresunet  --checkpoint_path /home/dsp/6DCM_Grasp/zuo/MSCG/logs/log_GraspNet_MSCG_MSGE_no_scm/minkuresunet_epoch19.tar  --resume  --dataset_root /media/ntfs/graspnet    --max_epoch 30  --Ori True --model GraspNet_MSCG_context_seed_global_high
CUDA_VISIBLE_DEVICES=0 python train.py --camera realsense --log_dir logs/log_GraspNet_MSCG_MSGE_no_scm --batch_size 4   --learning_rate 0.001  --model_name minkuresunet  --checkpoint_path /home/dsp/6DCM_Grasp/zuo/MSCG/logs/log_GraspNet_MSCG_MSGE_no_scm/minkuresunet_epoch19.tar  --resume  --dataset_root /media/ntfs/graspnet    --max_epoch 30   --model GraspNet_MSCG_context_seed_global_high

