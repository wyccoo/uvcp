# UVCP

### 1. Prepare python and CUDA：
   
```bash
python==3.8
CUDA==11.3
```
### 2. Install Pytorch:
```bash
  pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### 3. Install dependent environment:
```bash
  cd uvcp/
  pip install -r requirements.txt
  pip install -v -e .
  cd nuscenes-devkit-1.1.3/setup/
  pip install -v -e .
  cd ../../
```
### 4. Prepare dataset
```bash
  mkdir data/
```

   Download and unzip the dataset to the folder named data.
   
### 5. After downloading the dataset, generate a pkl file：
```bash
  python tools/create_data_bevdet_v2u.py
```
### 6. Test Model：
```bash
  python tools/test.py configs/UVCP/uvcpnet.py $checkpoint$ --eval map
```
