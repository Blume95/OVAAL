import os
from glob import glob
import numpy as np
from tool.build_dataset.utils import *

import cv2

ignore_class_label = 255

classes_to_labels = {
    0: ignore_class_label,
    1: ignore_class_label,
    2: ignore_class_label,
    3: ignore_class_label,
    4: ignore_class_label,
    5: ignore_class_label,
    6: ignore_class_label,
    7: 0,
    8: 1,
    9: ignore_class_label,
    10: ignore_class_label,
    11: 2,
    12: 3,
    13: 4,
    14: ignore_class_label,
    15: ignore_class_label,
    16: ignore_class_label,
    17: 5,
    18: ignore_class_label,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: ignore_class_label,
    30: ignore_class_label,
    31: 16,
    32: 17,
    33: 18,
    -1: ignore_class_label
}
index_label = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic_light',
    7: 'traffic_sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle'
}


def resize_(image_array, new_shape):
    new_img = cv2.resize(image_array, dsize=new_shape, interpolation=cv2.INTER_LINEAR)
    return new_img


if __name__ == "__main__":
    base_size = (1242, 375)
    org_label_dir = "/home/jing/Downloads/data_semantics/training/semantic"
    org_image_dir = "/home/jing/Downloads/data_semantics/training/image_2"
    org_label_path = glob(f"{org_label_dir}/*.png")
    org_image_path = glob(f"{org_image_dir}/*.png")
    assert len(org_label_path) == len(org_image_path)

    dst_dir = "/home/jing/Downloads/my_data/kitti_semantic"
    dst_images_dir = f"{dst_dir}/images"
    dst_label_dir = f"{dst_dir}/labels"
    dst_imageSets = f"{dst_dir}/ImageSets"

    os.makedirs(dst_imageSets, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)
    os.makedirs(dst_images_dir, exist_ok=True)

    for image_path in tqdm(org_image_path):
        image_ = cv2.imread(image_path, -1)
        image_new = resize_(image_, base_size)

        cv2.imwrite(f"{dst_images_dir}/{image_path.split('/')[-1]}", image_new)

    for label_path in tqdm(org_label_path):
        label = cv2.imread(label_path, -1)
        label_new = resize_(label, base_size)

        label_new_copy = label_new.copy()

        for index, new_index in classes_to_labels.items():
            label_new_copy[label_new == index] = new_index

        cv2.imwrite(f"{dst_label_dir}/{label_path.split('/')[-1]}", label_new_copy)

    label_names = [x.split('/')[-1] for x in org_label_path]

    random_num = np.random.rand(len(label_names))
    train_names = list(np.array(label_names)[random_num < 0.8])
    val_names = list(np.array(label_names)[random_num >= 0.8])

    write_txt(f"{dst_imageSets}/train.txt", train_names)
    write_txt(f"{dst_imageSets}/val.txt", val_names)

    create_ova_labels(dst_dir, label_names, index_label, ignore_class_label)
