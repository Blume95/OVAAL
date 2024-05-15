# import os
# dataset_path = "/home/jing/Downloads/my_data/cs"
# with open(os.path.join(dataset_path, "ImageSets/train.txt"), 'r') as f:
#     data = f.readlines()
#     data = [x.strip("\n") for x in data]
#
# print(data)
import re

# for i in range(10):
#     if i == 4:
#         pass
#         print(i)
#     else:
#         print(i)
import cv2
import numpy as np

# label = cv2.imread("/home/jing/Downloads/data_semantics/training/semantic/000000_10.png",
#                    -1)
# for i in range(33):
#     print(i)
#     new = np.zeros_like(label)
#     new[label==i] = 255
#     cv2.imshow("test", new)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
# print(np.random.rand(10)>0.5)
from glob import glob
import re
import cv2

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

# index_label = {
#     0: "sky",
#     1: "building",
#     2: "pole",
#     3: "road",
#     4: "sidewalk",
#     5: "tree",
#     6: "sign_symbol",
#     7: "fence",
#     8: "car",
#     9: "pedestrian",
#     10: "bicyclist"
# }

# a = glob(f"/home/jing/Downloads/ICRA_2024/ICRA_NEW/output/train_kitti_30_entropy_1.0_true/*/car/best_epoch_model.pt")
#
#
# def find_the_int_num(txt_string):
#     return list(map(int, txt_string.split("/")[-3][0]))[0]
#
#
# a.sort(key=find_the_int_num)
#
# print(a)

numbres = {x: 0 for x in index_label.keys()}
#
label_path = glob("/home/jing/Downloads/my_data/kitti_semantic/labels/*.png")
print(len(label_path))
for ele in label_path:
    label = cv2.imread(ele, -1)
    for index in numbres:
        a = label == index
        numbres[index] += np.sum(a)

new = {k: v for k, v in sorted(numbres.items(), key=lambda item: item[1])}
print(new)

# sign_symbol = glob("/home/jing/Downloads/my_data/cv/sign_symbol_labels/*.png")
# a = 0
# for e in sign_symbol:
#     label = cv2.imread(e, -1)
#     a += np.sum(label == 1)
# print(a)
# import pickle as pkl
#
# with open("/home/jing/Downloads/ICRA_2024/ICRA_NEW/output/train_cv_30_entropy_1.0_False(backup)/1_th/sign_symbol_queries.pkl",
#           'rb') as f:
#     data = pkl.load(f)
#
# for k, v in data.items():
#     print(k)
#     label = cv2.imread(f'/home/jing/Downloads/my_data/cv/sign_symbol_labels/{k}', -1)
#     print(np.unique(label))
#     axis_ = list(zip(v['y_coords'], v['x_coords']))
#     for x in axis_:
#         numbres[label[x]] += 1
#
# print(numbres)
