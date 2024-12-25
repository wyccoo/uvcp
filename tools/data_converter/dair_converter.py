# Copyright (c) OpenMMLab. All rights reserved.
import os
import re
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import pickle
import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox import points_cam2img
from mmdet3d.datasets import NuScenesDataset

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

def extract_vehicle_number(path_key, data):
    """
    从指定路径中提取vehicle的数字部分
    """
    if path_key in data:
        match = re.search(r"(\d+)", data[path_key])
        if match:
            return match.group(1)  # 返回匹配到的数字部分
    return None

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data
def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def rotation_matrix_to_quaternion(R):
    # R: 3x3 旋转矩阵

    # 计算四元数的标量部分 w
    w = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    
    # 计算四元数的 x, y, z 部分
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    
    # 返回四元数 (w, x, y, z)
    return np.array([w, x, y, z])

def obtain_sensor2top(cam_path, veh_Tr_velo_to_cam, inf_Tr_velo_to_cam, world2veh_lidar, world2inf_lidar, cam, numb):
    rect = np.eye(4)
    Trv2c = veh_Tr_velo_to_cam
    inf_Trv2c = inf_Tr_velo_to_cam
    world2veh_lidar = world2veh_lidar
    world2inf_lidar = world2veh_lidar

    veh_lidar2veh_cam = rect @ Trv2c
    sweep = {
        'data_path': cam_path,
        'type': cam,
        'sample_data_token': "",
        'sensor2ego_translation': veh_lidar2veh_cam[:3,3].T,
        'sensor2ego_rotation': rotation_matrix_to_quaternion(veh_lidar2veh_cam[:3,:3]),
        'ego2global_translation': world2veh_lidar[:3,3].T,
        'ego2global_rotation': rotation_matrix_to_quaternion(world2veh_lidar[:3,:3]),
        'timestamp': numb
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(world2veh_lidar[:3,:3]).T @ np.linalg.inv(veh_Tr_velo_to_cam[:3,:3]).T)
    T = (l2e_t_s @ e2g_r_s_mat[:3,:3].T + e2g_t_s) @ (
        np.linalg.inv(world2veh_lidar[:3,:3]).T @ np.linalg.inv(veh_Tr_velo_to_cam[:3,:3]).T)
    T -= world2veh_lidar[:3,3].T @ (np.linalg.inv(world2veh_lidar[:3,:3]).T @ np.linalg.inv(veh_Tr_velo_to_cam[:3,:3]).T
                  ) + veh_Tr_velo_to_cam[:3,3].T @ np.linalg.inv(veh_Tr_velo_to_cam[:3,:3]).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep

def obtain_sensor2top_world(cam_path, veh_Tr_velo_to_cam, inf_Tr_velo_to_cam, world2veh_lidar, world2inf_lidar, cam, numb):
    rect = np.eye(4)
    Trv2c = veh_Tr_velo_to_cam
    inf_Trv2c = inf_Tr_velo_to_cam
    world2veh_lidar = world2veh_lidar
    world2inf_lidar = world2veh_lidar

    veh_lidar2veh_cam = rect @ Trv2c
    veh_lidar2world = np.linalg.inv(world2veh_lidar)
    veh_lidar2inf_cam = rect @ inf_Trv2c @  world2inf_lidar @ veh_lidar2world
    sweep = {
        'data_path': cam_path,
        'type': cam,
        'sample_data_token': "",
        'sensor2ego_translation': veh_lidar2inf_cam[:3,3].T,
        'sensor2ego_rotation': rotation_matrix_to_quaternion(veh_lidar2inf_cam[:3,:3]),
        'ego2global_translation': world2inf_lidar[:3,3].T,
        'ego2global_rotation': rotation_matrix_to_quaternion(world2inf_lidar[:3,:3]),
        'timestamp': numb
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(world2inf_lidar[:3,:3]).T @ np.linalg.inv(inf_Tr_velo_to_cam[:3,:3]).T)
    T = (l2e_t_s @ e2g_r_s_mat[:3,:3].T + e2g_t_s) @ (
        np.linalg.inv(world2inf_lidar[:3,:3]).T @ np.linalg.inv(inf_Tr_velo_to_cam[:3,:3]).T)
    T -= world2inf_lidar[:3,3].T @ (np.linalg.inv(world2inf_lidar[:3,:3]).T @ np.linalg.inv(inf_Tr_velo_to_cam[:3,:3]).T
                  ) + inf_Tr_velo_to_cam[:3,3].T @ np.linalg.inv(inf_Tr_velo_to_cam[:3,:3]).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep

def get_info(root_path, split_items, extend_matrix=True):
    path = f'{root_path}/cooperative/data_info.json'
    data = load_json(path)
    path2 = f'{root_path}/vehicle-side/data_info.json'
    data2 = load_json(path2)
    info_list = []  # 存储所有生成的 info
    x1, y1 = -float('inf'), -float('inf')
    numb1 = extract_vehicle_number("vehicle_image_path", data[0])
    liness = []
    for sample in data:
        numb = extract_vehicle_number("vehicle_image_path", sample)
        numbinf = extract_vehicle_number("infrastructure_image_path", sample)
        if numb not in split_items:
            continue
        fold_name = f"{int(numb):06}.txt"
        label_path = f'{root_path}calib'
        data_label = os.path.join(label_path, fold_name)
        with open(data_label, 'r') as file:
            lines = file.readlines()
        P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                        ]).reshape([3, 4])
        P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                        ]).reshape([3, 4])
        P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                        ]).reshape([3, 4])
        P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                        ]).reshape([3, 4])
        if extend_matrix:
            P0 = _extend_matrix(P0)
            P1 = _extend_matrix(P1)
            P2 = _extend_matrix(P2)
            P3 = _extend_matrix(P3)
        R0_rect = np.array([
            float(info) for info in lines[4].split(' ')[1:10]
        ]).reshape([3, 3])
        if extend_matrix:
            rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
            rect_4x4[3, 3] = 1.
            rect_4x4[:3, :3] = R0_rect
        else:
            rect_4x4 = R0_rect

        veh_Tr_velo_to_cam = np.array([
            float(info) for info in lines[5].split(' ')[1:13]
        ]).reshape([3, 4])
        inf_Tr_velo_to_cam = np.array([
            float(info) for info in lines[6].split(' ')[1:13]
        ]).reshape([3, 4])
        world2veh_lidar = np.array([
            float(info) for info in lines[7].split(' ')[1:13]
        ]).reshape([3, 4])
        world2inf_lidar = np.array([
            float(info) for info in lines[8].split(' ')[1:13]
        ]).reshape([3, 4])
        lidar_path = sample["vehicle_pointcloud_path"]  # 使用 vehicle-side 的点云路径
        if extend_matrix:
            veh_Tr_velo_to_cam = _extend_matrix(veh_Tr_velo_to_cam)
            inf_Tr_velo_to_cam = _extend_matrix(inf_Tr_velo_to_cam)
            world2veh_lidar = _extend_matrix(world2veh_lidar)
            world2inf_lidar = _extend_matrix(world2inf_lidar)
        cs_record = {
            "translation": [0, 0, 0],  # 根据具体数据设置 lidar2ego_translation
            "rotation": [1, 0, 0, 0],  # 根据具体数据设置 lidar2ego_rotation
        }
        pose_record = {
            "translation": [0, 0, 0],  # 根据具体数据设置 ego2global_translation
            "rotation": [1, 0, 0, 0],  # 根据具体数据设置 ego2global_rotation
        }
        info = {
            "lidar_path": lidar_path,
            "token": "",  # 如果 token 不存在，则用默认值
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": int(numb),  # 如果 timestamp 不存在，则用默认值
        }
        camera_types = [
            'vehicle',
            'inf',
        ]
        for cam in camera_types:
            if cam == 'vehicle':
                cam_token = ""
                cam_path = f'{root_path}vehicle-side/image/{numb}.jpg'
                cam_intrinsic = P2[:3, :3]
                cam_info = obtain_sensor2top(cam_path, veh_Tr_velo_to_cam, inf_Tr_velo_to_cam, world2veh_lidar, world2inf_lidar, cam, numb)
                cam_info.update(cam_intrinsic=cam_intrinsic)
            else:
                cam_token = ""
                cam_path = f'{root_path}infrastructure-side/image/{numbinf}.jpg'
                cam_intrinsic = P3[:3, :3]
                cam_info = obtain_sensor2top_world(cam_path, veh_Tr_velo_to_cam, inf_Tr_velo_to_cam, world2veh_lidar, world2inf_lidar, cam, numbinf)
                cam_info.update(cam_intrinsic=cam_intrinsic)

            info['cams'].update({cam: cam_info})
        fold_name1 = f"{int(numb):06}.txt"
        label_path1 = f'{root_path}label2'
        data_label1 = os.path.join(label_path1, fold_name1)
        with open(data_label1, 'r') as file:
            lines = file.readlines()
        # info["gt_boxes"] = []
        # info["valid_flag"] = []
        # info["gt_names"] = []
        # info["gt_velocity"] = []
        gt_boxes = []
        valid_flag = []
        gt_names = []
        gt_velocity = []
        i=0
        for obj in lines:
            # 提取所需的字段
            fields = obj.split(' ')
            category = fields[0]

            # 后续 14 项是数值
            values = [float(info) for info in fields[1:]]
            truncated = int(values[0])  # 截断
            occluded =int(values[1])  # 遮挡
            alpha = values[2]  # 观察角度
            xmin, ymin, xmax, ymax = values[3:7]  # 2D 边界框 (xmin, ymin, xmax, ymax)
            h, w, l = values[7:10]  # 物体的尺寸 (高、宽、长)
            x, y, z = values[10:13]  # 物体的 3D 位置 (x, y, z)
            rotation_y = values[13]  # 物体的旋转角度
            gt_boxes.append([x, y, z, h, w, l, rotation_y])
            gt_names.append(category)
            if liness is None or i >= len(liness) or numb==numb1:
                gt_velocity.append([0, 0])
            else:
                xv = (float(lines[i].split(' ')[11])-float(liness[i].split(' ')[11]))/(float(numb1)-float(numb))
                yv = (float(lines[i].split(' ')[12])-float(liness[i].split(' ')[12]))/(float(numb1)-float(numb))
                gt_velocity.append([xv, yv])
            valid_flag.append(occluded==0 or occluded == 0)
            i += 1
        numb1=numb
        info["gt_boxes"] = np.array(gt_boxes)
        info["gt_names"] = np.array(gt_names)
        info["gt_velocity"] = np.array(gt_velocity)
        info["valid_flag"] = np.array(valid_flag)
        info["scene_token"] = 000000
        liness = lines
        info_list.append(info)
    # info = {
    #     'lidar_path': lidar_path,
    #     'token': sample['token'],
    #     'sweeps': [],
    #     'cams': dict(),
    #     'lidar2ego_translation': cs_record['translation'],
    #     'lidar2ego_rotation': cs_record['rotation'],
    #     'ego2global_translation': pose_record['translation'],
    #     'ego2global_rotation': pose_record['rotation'],
    #     'timestamp': sample['timestamp'],
    # }
    # # for item in split_items:
    return info_list


def create_dair_infos(root_path,
                          info_prefix,
                          version='v1.14-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    # from nuscenes.nuscenes import NuScenes
    # nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # from nuscenes.utils import splits
    # available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini', 'v1.14-trainval']
    # assert version in available_vers
    # if version == 'v1.0-trainval':
    #     train_scenes = splits.train
    #     val_scenes = splits.val
    # elif version == 'v1.0-test':
    #     train_scenes = splits.test
    #     val_scenes = []
    # elif version == 'v1.0-mini':
    #     train_scenes = splits.mini_train
    #     val_scenes = splits.mini_val
    # elif version == 'v1.14-trainval':
    #     train_scenes = splits.trains
    #     val_scenes = splits.vals
    # else:
    #     raise ValueError('unknown')

    import os
    import json
    import shutil

    # 加载 JSON 数据
    json_file_path = f'{root_path}/split_datas/cooperative-split-data.json'  # 替换为你的 JSON 文件路径
    # output_directory = "output"  # 数据划分输出的根目录

    with open(json_file_path, "r") as f:
        data = json.load(f)

    # 获取 cooperative_split 数据
    cooperative_split = data["cooperative_split"]

    # 定义数据集文件夹名称
    splits = ["train", "val", "test", "test_A"]

    # 假设你的实际文件存储在一个目录中
    source_directory = "source_files"  # 替换为实际文件存储目录

    # 创建目标文件夹并移动文件
    for split in splits:
        split_items = cooperative_split[split]
        # target_directory = os.path.join(output_directory, split)
        # os.makedirs(target_directory, exist_ok=True)  # 创建文件夹
        pkl_file_path = os.path.join(root_path, f"{split}.pkl")
        infos = get_info(root_path, cooperative_split[split])
        metadata = dict(version=version)
        data = dict(infos=infos, metadata=metadata)
        with open(pkl_file_path, "wb") as f:  # 使用 "wb" 模式写入二进制文件
            pickle.dump(data, f)
        



