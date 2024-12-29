import pickle

# 打开.pkl文件并读取数据
with open('/workspace/DATA/datasetWYC/bevdetv2-nuscenes_infos_train.pkl', 'rb') as pkl_file:
    data = pickle.load(pkl_file)

# 打开.txt文件并将数据写入
with open('dataWYC.txt', 'w') as txt_file:
    txt_file.write(str(data))

print("Pickle文件已成功转换为文本文件。")
