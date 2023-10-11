# Installation

## Get the code
```bash
git clone git@github.com:chisarie/contact_graspnet.git -b inference
git clone git@github.com:chisarie/uois.git -b inference
```

## Original Installation
```bash
cd contact_graspnet
conda env create -f contact_graspnet_env.yml
conda activate contact_graspnet_env
pip install scikit-image torch==1.5.1 torchvision==0.6.1
pip install -e ../uois
```

## Installation in a 30 series gpu

- From https://github.com/NVlabs/contact_graspnet/issues/19#issuecomment-1179489804

```bash
conda create --name contact_graspnet python=3.8
conda activate contact_graspnet
conda install -c conda-forge cudatoolkit=11.3
conda install -c conda-forge cudnn=8.2
pip install tensorflow==2.5 tensorflow-gpu==2.5 opencv-python-headless pyyaml==5.4.1 pyrender tqdm mayavi pyqt5

pip install scikit-image==0.19 torch torchvision
pip install -e ../uois
```

## Post-Install
After you have successfully create a env, don't forget to recompile `pointnet2 tf_ops`:
```bash
sh compile_pointnet_tfops.sh
```
