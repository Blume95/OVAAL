import pickle as pkl
import cv2

sample = "/home/jing/Downloads/ICRA_2024/ICRA_NEW/output/train_pixelpick_cv_50_entropy_11.0_backup/9_th/queries.pkl"
with open(sample, 'rb') as f:
    data = pkl.load(f)

for k,v in data.items():
    print(len(v['x_coords']))

# print(sorted(data['label_distribution'].items(), key=lambda x: x[1]))

# tem_dict = dict()
#
# for k, v in data.items():
#     path = k.replace("/media/expleo/B4BCA1DFBCA19BFC/Jing_thesis/OVAAL/data/",
#                      "/home/jing/Downloads/my_data/cityscapes/").replace("/leftImg8bit/", "/gtFine/").replace(
#         "leftImg8bit.png", 'gtFine_labelIds.png')
#
#     label = cv2.imread(path, -1)
#     x_coords = v['x_coords']
#     y_coords = v['y_coords']
#
#     for index, x in enumerate(x_coords):
#         y = y_coords[index]
#         if tem_dict.get(label[y, x]) is None:
#             tem_dict[label[y, x]] = 1
#         else:
#             tem_dict[label[y, x]] += 1
#
# print(sorted(tem_dict.items(), key=lambda x: x[1]))
# "[(17, 26251), (16, 32784), (12, 33001), (6, 36381), (15, 39003), (14, 40280), (18, 69317), (7, 71075), (3, 80076), (9, 106449), (4, 118098), (10, 142095), (5, 147189), (11, 163890), (1, 349337), (13, 559009), (19, 691092), (0, 794305), (8, 810603), (2, 1342067)]"
# "[(16, 16371), (17, 20163), (15, 27895), (14, 30564), (12, 35735), (6, 49388), (18, 82208), (7, 116601), (3, 133803), (4, 185114), (9, 194729), (10, 198692), (11, 205731), (5, 319869), (13, 380178), (0, 621469), (1, 680608), (8, 973556), (2, 1379826)]"