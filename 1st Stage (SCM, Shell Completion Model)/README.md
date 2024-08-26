## Demo of SCM Method Flow

![SCM](https://github.com/user-attachments/assets/d190d81b-af9f-458e-ad11-5b6802e4be0f)

## Requirements
- torchvision
- tensorboardX
- librosa==0.7.2
- inflect==0.2.5
- Unidecode==1.0.22
- pillow
- configargparse

## Installation
Install packages via Pip.
```bash
pip install -r requirements.txt
```

## Training and Testing
Training examples are shown in [command_train.sh](command_train.sh). `--name` and `--config` should be specified according to your settings. You can use TensorBoard to visualize training process.

Testing examples are shown in [test.py](test.py), which contains result evaluation of score.
