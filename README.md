# UVCP-DAIR

1. CUDA(11.3) installation：
   
```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
  sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda-repo-ubuntu1804-11-3-local_11.3.1-465.19.01-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu1804-11-3-local_11.3.1-465.19.01-1_amd64.deb
  sudo apt-key add /var/cuda-repo-ubuntu1804-11-3-local/7fa2af80.pub
  sudo apt-get update
  sudo apt-get -y install cuda
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
   Download v2u-coo dataset data folder

5. After downloading the dataset, generate a pkl file：
```bash
 python 
  python tools/create_data_bevdet_v2u.py
```
6. Test Model：
```bash
  python tools/test.py configs/UVCP/uvcpnet.py $checkpoint$ --eval map
```
