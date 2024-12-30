# UVCP-DAIR

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
  pip install -r requirements.txt
  pip install -v -e .
  cd nuscenes-devkit-1.1.3/setup/
  pip install -v -e .
  cd ../
  git clone https://github.com/klintan/pypcd.git
  cd pypcd
  python setup.py install
  cd ../
```
### 4. Prepare dataset

Download the dataset to the folder named data and form the following structure.\
Download the dataset from https://thudair.baai.ac.cn/coop-dtest

  ```

### Download the dataset to the folder named data and form the following structure
├── cooperative-vehicle-infrastructure      # DAIR-V2X-C
    ├── infrastructure-side             # DAIR-V2X-C-I
        ├── image		    
            ├── {id}.jpg
        ├── velodyne                
            ├── {id}.pcd           
        ├── calib                 
            ├── camera_intrinsic            
                ├── {id}.json     
            ├── virtuallidar_to_world   
                ├── {id}.json      
            ├── virtuallidar_to_camera  
                ├── {id}.json      
        ├── label	
            ├── camera                  # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in image based on image frame time
                ├── {id}.json
            ├── virtuallidar            # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
                ├── {id}.json
        ├── data_info.json              # Relevant index information of Infrastructure data
    ├── vehicle-side                    # DAIR-V2X-C-V
        ├── image		    
            ├── {id}.jpg
        ├── velodyne             
            ├── {id}.pcd           
        ├── calib                 
            ├── camera_intrinsic   
                ├── {id}.json
            ├── lidar_to_camera   
                ├── {id}.json
            ├── lidar_to_novatel  
                ├── {id}.json
            ├── novatel_to_world   
                ├── {id}.json
        ├── label	
            ├── camera                  # Labeled data in Vehicle LiDAR Coordinate System fitting objects in image based on image frame time
                ├── {id}.json
            ├── lidar                   # Labeled data in Vehicle LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
                ├── {id}.json
        ├── data_info.json              # Relevant index information of the Vehicle data
    ├── cooperative                     # Coopetative Files
        ├── label_world                 # Vehicle-Infrastructure Cooperative (VIC) Annotation files
            ├── {id}.json           
        ├── data_info.json              # Relevant index information combined the Infrastructure data and the Vehicle data
```
After forming the dataset, run the following command
```bash
cp data/data_info.json data/cooperative-vehicle-infrastructure/cooperative/
```


Transform DAIR-V2X

Run the following command to convert DAIR-V2X
```python
python tools/data_converter/dair_vic2kitti_2.py
```

### 5. After downloading the dataset, generate pkl files：
```bash
  python tools/create_dair.py
```
### 6. Test Model：
```bash
  python tools/test.py configs/uvcp/uvcp-dair.py $checkpoint$ --eval map
```
