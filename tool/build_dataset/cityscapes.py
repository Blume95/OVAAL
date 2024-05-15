import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm

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
frequency = {0: 24.299563467502594, 1: 3.60712930560112, 2: 12.082608044147491, 3: 0.4063664935529232,
             4: 0.5820929072797298, 5: 0.7717564236372709, 6: 0.20481094252318144, 7: 0.36028504837304354,
             8: 8.916863799095154, 9: 0.735138263553381, 10: 1.805989071726799, 11: 0.9288511238992214,
             12: 0.12774631613865495, 13: 4.758162796497345, 14: 0.17884725239127874, 15: 0.1690849312581122,
             16: 0.1574836322106421, 17: 0.068853102857247, 18: 0.29661229345947504}


def process_label_image(value, image_dst, label_dst, factor_):
    if factor_ == 4:
        mode = "train"
    if factor_ == 2:
        mode = "val"

    new_name_list = []
    for index, image_path in tqdm(enumerate(value['image'])):
        image = cv2.imread(image_path, -1)
        down_size_image = cv2.resize(image, (int(image.shape[1] / factor_), int(image.shape[0] / factor_)),
                                     interpolation=cv2.INTER_LINEAR)

        label_path = image_path.replace("/leftImg8bit/", "/gtFine/").replace("leftImg8bit.png", 'gtFine_labelIds.png')
        label = cv2.imread(label_path, -1)
        down_size_label = cv2.resize(label, (int(label.shape[1] / factor_), int(label.shape[0] / factor_)),
                                     interpolation=cv2.INTER_LINEAR)
        down_size_label_flatten = down_size_label.flatten()
        for i in range(len(down_size_label_flatten)):
            down_size_label_flatten[i] = classes_to_labels[down_size_label_flatten[i].item()]

        down_size_label = down_size_label_flatten.reshape((down_size_label.shape[0], down_size_label.shape[1]))

        cv2.imwrite(os.path.join(image_dst, f"{mode}_{index:06}.png"), down_size_image)
        cv2.imwrite(os.path.join(label_dst, f"{mode}_{index:06}.png"), down_size_label)

        new_name_list.append(f"{mode}_{index:06}.png")

    return new_name_list


def create_ova_labels(root_path, label_names_list):
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


def write_txt(txt_path, txt):
    with open(txt_path, 'w') as f:
        for item in txt:
            f.write("%s\n" % item)


def build_cityscapes(org_path, dst_path, label_dir="gtFine", image_dir="leftImg8bit"):
    train_image_dir = os.path.join(org_path, image_dir + "/train")
    val_image_dir = os.path.join(org_path, image_dir + "/val")

    train_label_dir = os.path.join(org_path, label_dir + "/train")
    val_label_dir = os.path.join(org_path, label_dir + "/val")

    train_image_list = glob(f"{train_image_dir}/*/*.png")
    val_image_list = glob(f"{val_image_dir}/*/*.png")

    train_label_list = glob(f"{train_label_dir}/*/*labelIds.png")
    val_label_list = glob(f"{val_label_dir}/*/*labelIds.png")

    assert len(train_image_list) == len(train_label_list), "label != image"
    assert len(val_image_list) == len(val_label_list), "label != image"

    image_dst = os.path.join(dst_path, "images")
    label_dst = os.path.join(dst_path, "labels")
    imageSets = os.path.join(dst_path, "ImageSets")

    os.makedirs(imageSets, exist_ok=True)
    os.makedirs(label_dst, exist_ok=True)
    os.makedirs(image_dst, exist_ok=True)

    # train down sample with factor 4 , val down sampling with factor 2, not interested class set as 255
    tem_dict = {"train": {"image": train_image_list, "label": train_label_list},
                "val": {"image": val_image_list, "label": val_label_list}}

    train_names = process_label_image(tem_dict['train'], image_dst, label_dst, 4)
    val_names = process_label_image(tem_dict['val'], image_dst, label_dst, 2)

    write_txt(os.path.join(imageSets, "train.txt"), train_names)
    write_txt(os.path.join(imageSets, "val.txt"), val_names)

    create_ova_labels(dst_path, train_names + val_names)


if __name__ == "__main__":
    your_cityscapes_dataset = "/home/jing/Downloads/my_data/cityscapes"
    your_dst_dataset = "/home/jing/Downloads/my_data/cs"

    # build_cityscapes(your_cityscapes_dataset, your_dst_dataset)
    # some_path = "/home/jing/Downloads/my_data/cs"
    # label_names_ = glob(f"{some_path}/labels/*.png")
    # label_names_ = [x.split("/")[-1] for x in label_names_]
    # create_ova_labels(some_path, label_names_)
