import os.path
from tool.build_dataset.utils import *
import cv2
import numpy as np
from PIL import Image
from glob import glob

ignore_class_label = 255

index_label = {
    0: "sky",
    1: "building",
    2: "pole",
    3: "road",
    4: "sidewalk",
    5: "tree",
    6: "sign_symbol",
    7: "fence",
    8: "car",
    9: "pedestrian",
    10: "bicyclist",
}
label_index = {v: k for k, v in index_label.items()}

color_label = {
    (64, 128, 64): 'animal',
    (192, 0, 128): 'archway',
    (0, 128, 192): 'bicyclist',
    (0, 128, 64): 'bridge',
    (128, 0, 0): 'building',
    (64, 0, 128): 'car',
    (64, 0, 192): 'cartluggagepram',
    (192, 128, 64): 'child',
    (192, 192, 128): 'pole',
    (64, 64, 128): 'fence',
    (128, 0, 192): 'lanemkgsdriv',
    (192, 0, 64): 'lanemkgsnondriv',
    (128, 128, 64): 'misc_text',
    (192, 0, 192): 'motorcyclescooter',
    (128, 64, 64): 'othermoving',
    (64, 192, 128): 'parkingblock',
    (64, 64, 0): 'pedestrian',
    (128, 64, 128): 'road',
    (128, 128, 192): 'road shoulder',
    (0, 0, 192): 'sidewalk',
    (192, 128, 128): 'sign_symbol',
    (128, 128, 128): 'sky',
    (64, 128, 192): 'suvpickuptruck',
    (0, 0, 64): 'trafficcone',
    (0, 64, 64): 'trafficlight',
    (192, 64, 128): 'train',
    (128, 128, 0): 'tree',
    (192, 128, 192): 'truck_bus',
    (64, 0, 64): 'tunnel',
    (192, 192, 0): 'vegetationmisc',
    (0, 0, 0): 'void',
    (64, 192, 0): 'wall'
}

label_color = {v: k for k, v in color_label.items()}

frequency = {0: 13.247638940811157, 1: 18.95792782306671, 2: 0.7164792157709599, 3: 23.31564724445343,
             4: 2.891908213496208,
             5: 7.530459016561508, 6: 0.057672226103022695, 7: 1.0449297726154327, 8: 4.3109603226184845,
             9: 0.6148428190499544, 10: 0.26557783130556345}

# label_dir = "/home/jing/Downloads/my_data/cv/labels"
# dst_dir = "/home/jing/Downloads/my_data/cv/labels_new"
# os.makedirs(dst_dir, exist_ok=True)
# labels_path = glob(f"{label_dir}/*.png")
# label_names_list_ = [x.split("/")[-1] for x in labels_path]
#
#
# for label_path in labels_path:
#     label_png = np.array(Image.open(label_path).convert("RGB"))
#     new_label_png = np.ones((label_png.shape[0], label_png.shape[1]), dtype=np.uint8) * ignore_index
#     for index, label in index_label.items():
#         color_rgb = label_color[label]
#         matches = (label_png == np.array(color_rgb)).all(axis=2)
#         new_label_png[matches] = index
#
#     cv2.imwrite(os.path.join(dst_dir, label_path.split("/")[-1]), new_label_png)


# test = cv2.imread(f"{label_dir}/0006R0_f01980_L.png", -1)
# print(test.shape)
# cv2.imshow("test", test)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import re
#
# txt_path = "/home/jing/Downloads/ICRA_2024/ICRA_NEW/tool/build_dataset/cv_val.txt"
# dst_path = "/home/jing/Downloads/my_data/cv/ImageSets/val.txt"
# with open(txt_path, 'r') as f:
#     _txt = f.readlines()
# png_name_list = []
# print(_txt)
#
# for line in _txt:
#     matches = re.findall('test/(.*?png)', line)
#     if len(matches) > 0:
#         png_name_list.append(matches[0])
#
#
# def write_txt(txt_path_, txt_):
#     with open(txt_path_, 'w') as f:
#         for item in txt_:
#             f.write("%s\n" % item)
#
# write_txt(dst_path,png_name_list)


# txt_path = "/home/jing/Downloads/my_data/cv/ImageSets/train.txt"
# with open(txt_path, 'r') as f:
#     _txt = f.readlines()
#     _txt = [x.strip("\n") for x in _txt]
#
# for line in _txt:
#
#     image = cv2.imread(f"/home/jing/Downloads/my_data/cv/labels/{line}", -1)
#
#     image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)),
#                        interpolation=cv2.INTER_LINEAR)
#     print(image.shape)
#     cv2.imwrite(f"/home/jing/Downloads/my_data/cv/labels/{line}", image)

from tqdm import tqdm

# def create_ova_labels(root_path, label_names_list):
#     for label_name in tqdm(label_names_list):
#         label = cv2.imread(os.path.join(root_path, f"labels/{label_name}"), -1)
#         for index, class_name in index_label.items():
#             ova_dir = os.path.join(root_path, f"{class_name}_labels")
#             if os.path.exists(ova_dir):
#                 "Do nothing"
#             else:
#                 os.mkdir(ova_dir)
#
#             ignore_area = label == ignore_class_label
#             ova_area = label == index
#             label_tem = np.zeros_like(label)
#             label_tem[ova_area] = 1
#             label_tem[ignore_area] = ignore_class_label
#             cv2.imwrite(os.path.join(ova_dir, label_name), label_tem)


#
#
# create_ova_labels("/home/jing/Downloads/my_data/cv", label_names_list_)

# path = "/home/jing/Downloads/my_data/cv/labels"
# label_path_list = glob(f"{path}/*_L.png")
# # print(label_path_list)
# for label_path in label_path_list:
#     # label = cv2.imread(label_path, -1)
#     # print(label_path.replace("_L", ""))
#     # cv2.imwrite(label_path.replace("_L", ""), label)
#     print(label_path)
#     os.remove(label_path)


if __name__ == "__main__":
    root_path = "/home/jing/Downloads/my_data/cv"
    txt_train = f"{root_path}/ImageSets/train.txt"
    txt_val = f"{root_path}/ImageSets/val.txt"

    org_data_dir = "/home/jing/Downloads/my_data/camvid/CamVid"
    train_image_dir = f"{org_data_dir}/train"
    val_image_dir = f"{org_data_dir}/val"

    train_label_dir = f"{org_data_dir}/train_labels"
    val_label_dir = f"{org_data_dir}/val_labels"

    train_files_name = read_txt(txt_train)
    val_files_name = read_txt(txt_val)
    names_sum = train_files_name + val_files_name

    dst_images_dir = f"{root_path}/images"
    dst_labels_dir = f"{root_path}/labels"

    image_sum = glob(f"{train_image_dir}/*.png") + glob(f"{val_image_dir}/*.png")
    label_sum = glob(f"{train_label_dir}/*.png") + glob(f"{val_label_dir}/*.png")

    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    # train image 1/2 size, label from 3 channels to 1 channel, create ova labels , remove _L in label name

    for image_path in tqdm(image_sum):
        if image_path.split("/")[-1] in names_sum:
            image = cv2.imread(image_path, -1)
            if image_path.split("/")[-1] in train_files_name:
                image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)),
                                   interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(f"{dst_images_dir}/{image_path.split('/')[-1]}", image)

    for label_path in tqdm(label_sum):
        if label_path.split("/")[-1].replace("_L", "") in names_sum:
            label_png = np.array(Image.open(label_path).convert("RGB"))
            if label_path.split("/")[-1].replace("_L", "") in train_files_name:
                label_png = cv2.resize(label_png,
                                       (int(label_png.shape[1] / 2), int(label_png.shape[0] / 2)),
                                       interpolation=cv2.INTER_LINEAR)
            new_label_png = np.ones((label_png.shape[0], label_png.shape[1]), dtype=np.uint8) * ignore_class_label
            for index, label in index_label.items():
                color_rgb = label_color[label]
                matches = (label_png == np.array(color_rgb)).all(axis=2)
                new_label_png[matches] = index

            print(np.unique(new_label_png))

            cv2.imwrite(f"{dst_labels_dir}/{label_path.split('/')[-1].replace('_L', '')}", new_label_png)

    for label_name in tqdm(names_sum):
        label = cv2.imread(os.path.join(root_path, f"labels/{label_name}"), -1)
        for index, class_name in index_label.items():
            ova_dir = os.path.join(root_path, f"{class_name}_labels")
            if os.path.exists(ova_dir):
                "Do nothing"
            else:
                os.mkdir(ova_dir)

            ignore_area = label == ignore_class_label
            ova_area = label == index
            label_tem = np.zeros_like(label)
            label_tem[ova_area] = 1
            label_tem[ignore_area] = ignore_class_label
            cv2.imwrite(os.path.join(ova_dir, label_name), label_tem)
