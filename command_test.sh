
CUDA_VISIBLE_DEVICES=0 python test.py --camera realsense --dump_dir  logs/log_GraspNet_MSCG_MSGE_no_scm/dump_18_test_CD  --dataset_root /media/ntfs/graspnet  --checkpoint_path  /home/dsp/6DCM_Grasp/zuo/MSCG/logs/log_GraspNet_MSCG_MSGE_no_scm/minkuresunet_epoch18.tar  --batch_size 4   --eval --collision_thresh 0.01 --split test --model GraspNet_MSCG_context_seed_global_high

