import argparse
import os
from gen_kitti.label_coopcoord_to_cameracoord import gen_veh_lidar2veh_cam
from gen_kitti.label_json2kitti import json2kitti, rewrite_label, label_filter
from tools.data_converter.gen_kitti.gen_calib2kitti_coop import gen_calib2kitti_coop
from gen_kitti.gen_ImageSets_from_split_data import gen_ImageSet_from_coop_split_data
from tools.data_converter.gen_kitti.utils import pcd2bin

parser = argparse.ArgumentParser("Generate the Kitti Format Data")
parser.add_argument("--source-root", type=str, default="data/cooperative-vehicle-infrastructure", help="Raw data root about DAIR-V2X.")
parser.add_argument(
    "--target-root",
    type=str,
    default="data/dair_vic_kitti_format",
    help="The data root where the data with kitti format is generated",
)
parser.add_argument(
    "--split-path",
    type=str,
    default="data/split_datas/cooperative-split-data.json",
    help="Json file to split the data into training/validation/testing.",
)
parser.add_argument("--label-type", type=str, default="lidar", help="label type from ['lidar', 'camera']")
parser.add_argument("--sensor-view", type=str, default="vehicle", help="Sensor view from ['infrastructure', 'vehicle']")
parser.add_argument("--no-classmerge", action="store_true", help="Not to merge the four classes [Car, Truck, Van, Bus] into one class [Car]")
parser.add_argument("--temp-root", type=str, default="./tmp_file", help="Temporary intermediate file root.")


def mdkir_kitti(target_root):
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    os.system("mkdir -p %s/training" % target_root)
    os.system("mkdir -p %s/training/calib" % target_root)
    os.system("mkdir -p %s/training/label2" % target_root)
    os.system("mkdir -p %s/training/velodyne" % target_root)
    os.system("mkdir -p %s/training/image_2" % target_root)
    os.system("mkdir -p %s/training/image_3" % target_root)
    os.system("mkdir -p %s/training//velodyne_inf" % target_root)
    os.system("mkdir -p %s/testing" % target_root)
    os.system("mkdir -p %s/ImageSets" % target_root)


def rawdata_copy(source_root, target_root):
    os.system("cp -r %s/vehicle-side/image/* %s/training/image_2" % (source_root, target_root))
    os.system("cp -r %s/infrastructure-side/image/* %s/training/image_3" % (source_root, target_root))
    os.system("cp -r %s/vehicle-side/velodyne/* %s/training/velodyne" % (source_root, target_root))
    os.system("cp -r %s/infrastructure-side/velodyne/* %s/training/velodyne_inf" % (source_root, target_root))

def copy():
    os.system("cp -r data/dair_vic_kitti_format/training/calib data/cooperative-vehicle-infrastructure")
    os.system("cp -r data/dair_vic_kitti_format/training/label2 data/cooperative-vehicle-infrastructure")

# def kitti_pcd2bin(target_root):
#     pcd_dir = os.path.join(target_root, "training/velodyne_inf")
#     fileList = os.listdir(pcd_dir)
#     for fileName in fileList:
#         if "010937.pcd" in fileName:
#             pcd_file_path = pcd_dir + "/" + fileName
#             bin_file_path = pcd_dir + "/" + fileName.replace(".pcd", ".bin")
#             pcd2bin(pcd_file_path, bin_file_path)


if __name__ == "__main__":
    print("================ Start to Convert ================")
    args = parser.parse_args()
    source_root = args.source_root
    target_root = args.target_root

    print("================ Start to Copy Raw Data ================")
    mdkir_kitti(target_root)
    # rawdata_copy(source_root, target_root)
    # Preprocess the point cloud
    # kitti_pcd2bin(target_root)

    print("================ Start to Generate Label ================")
    temp_root = args.temp_root
    label_type = args.label_type
    no_classmerge = args.no_classmerge
    # os.system("mkdir -p %s" % temp_root)
    # os.system("rm -rf %s/*" % temp_root)

    ## Transform LABEL from world coord into vehicle camera coord
    gen_veh_lidar2veh_cam(source_root, temp_root, label_type=label_type)


    json_root = os.path.join(temp_root, "label", label_type)
    kitti_label_root = os.path.join(target_root, "training/label2")
    json2kitti(json_root, kitti_label_root)
    if not no_classmerge:
        rewrite_label(kitti_label_root)
    label_filter(kitti_label_root)

    os.system("rm -rf %s" % temp_root)


    print("================ Start to Generate Calibration Files ================")
    sensor_view = args.sensor_view

    #Obtain CALIB from both Vehicle and Infrastructure side
    gen_calib2kitti_coop(source_root, target_root, label_type=label_type)

    print("================ Start to Generate ImageSet Files ================")
    split_json_path = args.split_path
    ImageSets_path = os.path.join(target_root, "ImageSets")
    copy()
    # gen_ImageSet_from_split_data(ImageSets_path, split_json_path, sensor_view)
    gen_ImageSet_from_coop_split_data(ImageSets_path, split_json_path, sensor_view)