import re
import cv2
from tqdm import tqdm
import os
import numpy as np


def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        _txt = f.readlines()
        _txt = [x.strip("\n") for x in _txt]

    return _txt


def write_txt(txt_path_, txt_):
    with open(txt_path_, 'w') as f:
        for item in txt_:
            f.write("%s\n" % item)


def find_all_png_names(_txt, cond: str):
    png_name_list = []
    for line in _txt:
        matches = re.findall(cond, line)
        if len(matches) > 0:
            png_name_list.append(matches[0])

    return png_name_list


def create_ova_labels(root_path, label_names_list, index_label, ignore_class_label):
    for label_name in tqdm(label_names_list):
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
