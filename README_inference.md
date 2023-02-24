# Installation

```bash
git clone git@github.com:chisarie/contact_graspnet.git@inference
git clone git@github.com:chrisdxie/uois.git

cd contact_graspnet
conda env create -f contact_graspnet_env.yml
pip install scikit-image torch==1.5.1 torchvision==0.6.1
pip install -e ../uois
```
