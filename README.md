# SC-MsGNet

## Demonstration of SC-MsGNet Method Flow
![論文方法流程圖](https://github.com/user-attachments/assets/54376919-9b0a-4a16-97a9-91652f751025)

Paper: 

## Requirements
- Python 3
- PyTorch 1.8
- Open3d 0.8
- TensorBoard 2.3
- NumPy
- SciPy
- Pillow
- tqdm
- MinkowskiEngine

## Installation
Get the code.
```bash
git clone https://github.com/Kezuoxun/SC-MsGNet.git
cd graspnet-sc-msgnet
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```
For MinkowskiEngine, please refer https://github.com/NVIDIA/MinkowskiEngine
## Point level Graspness Generation
Point level graspness label are not included in the original dataset, and need additional generation. Make sure you have downloaded the orginal dataset from [GraspNet](https://graspnet.net/). The generation code is in [dataset/generate_graspness.py](dataset/generate_graspness.py).
```bash
cd dataset
python generate_graspness.py --dataset_root /data3/graspnet --camera_type kinect
```

## Simplify dataset
original dataset grasp_label files have redundant data,  We can significantly save the memory cost. The code is in [dataset/simplify_dataset.py](dataset/simplify_dataset.py)
```bash
cd dataset
python simplify_dataset.py --dataset_root /data3/graspnet
```

## Training and Testing
Training examples are shown in [command_train.sh](command_train.sh). `--dataset_root`, `--camera` and `--log_dir` should be specified according to your settings. You can use TensorBoard to visualize training process.

Testing examples are shown in [command_test.sh](command_test.sh), which contains inference and result evaluation. `--dataset_root`, `--camera`, `--checkpoint_path` and `--dump_dir` should be specified according to your settings. Set `--collision_thresh` to -1 for fast inference.

## Results
### Results  of my method#1 (CA-MsGE) results have collision detection.

Evaluation results on Realsense camera:
|         |              |         Seen         |        |           |        Similar          |        |            |         Novel          ||
|:------: |:----------------:|:----------------:|:------:|:----------------:|:----------------:|:------:|:----------------:|:----------------:|:----------------:|
|         | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> |
| w/o CD  | 74.47  | 84.83            | 71.12           | 62.28  | 74.53            | 55.25            | 25.91  | 32.10            | 14.07             |
|     CD  | 76.58  | 87.53            | 72.61            | 64.42  | 77.32            | 56.61            | 27.96  | 33.57            | 14.34             |

Results  of my method#1 (CA-MsGE) results have collision detection.


### Results  of my method#2 (MsGFF) results have collision detection.
Evaluation results on Realsense camera:
|         |              |         Seen         |        |           |        Similar          |        |            |         Novel          ||
|:------: |:----------------:|:----------------:|:------:|:----------------:|:----------------:|:------:|:----------------:|:----------------:|:----------------:|
|         | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> |
| w/o CD  | 73.93  | 83.54            | 71.03           | 60.98  | 72.79            | 54.19            | 24.70  | 30.67            | 13.38             |
|     CD  | 76.21  | 86.49            | 72.67            | 63.47  | 76.05            | 55.91            | 26.31  | 32.62            | 13.95             |


## Troubleshooting
If you meet the torch.floor error in MinkowskiEngine, you can simply solve it by changing the source code of MinkowskiEngine: 
MinkowskiEngine/utils/quantization.py 262，from discrete_coordinates =_auto_floor(coordinates) to discrete_coordinates = coordinates
## Acknowledgement
My code is mainly based on Graspnet-baseline  https://github.com/graspnet/graspnet-baseline.
