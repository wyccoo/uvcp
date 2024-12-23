# UVCP

1. CUDA(11.3) installation：
   
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
sudo sh cuda_11.3.1_465.19.01_linux.run
```
2. Install Pytorch:
```bash
  pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
3. Install dependent environment:
```bash
  pip install -r requirements.txt
  pip install -v -e .
  cd nuscenes-devkit-1.1.3/setup/
  pip install -v -e .
```
4. Prepare dataset

   Download the dataset to the folder named data
   
6. After downloading the dataset, generate a pkl file：
```bash
  python tools/create_data_bevdet_v2u.py
```
6. Test Model：
```bash
  python tools/test.py configs/UVCP/uvcpnet.py $checkpoint$ --eval map
```
