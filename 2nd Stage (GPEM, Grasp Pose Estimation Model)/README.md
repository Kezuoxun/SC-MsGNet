## Demo of GPEM Method Flow

![GPEM](https://github.com/user-attachments/assets/663efcac-0b16-4863-8546-e32268e4b806)

## Demo of Method 1 (CA-MsGE)  Flow
![CA-MsGE](https://github.com/user-attachments/assets/337578ba-02c8-414e-8ad9-78673fce680b)

## Demo of Method 2 (MsGFF)  Flow
![MSGFF](https://github.com/user-attachments/assets/8fd871b2-42b4-449d-841f-c9b8fff559ec)

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
git clone https://github.com/rhett-chen/graspness_implementation.git
cd graspnet-graspness
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


## Troubleshooting
If you meet the torch.floor error in MinkowskiEngine, you can simply solve it by changing the source code of MinkowskiEngine: 
MinkowskiEngine/utils/quantization.py 262，from discrete_coordinates =_auto_floor(coordinates) to discrete_coordinates = coordinates

## Acknowledgement
My code is mainly based on: 
1. Graspnet-baseline  https://github.com/graspnet/graspnet-baseline
2. Graspness_implementation https://github.com/rhett-chen/graspness_implementation.git
3. Scale-Balanced-Grasp  https://github.com/mahaoxiang822/Scale-Balanced-Grasp.git .
