import torch
import torchvision.transforms.functional as TF
from glob import glob
from PIL import Image

input_image_dir = "/home/jing/Downloads/my_data/kitti_semantic/images"
images_path = glob(f"{input_image_dir}/*.png")

#
cnt = 0
fst_moment = torch.empty(3)
snd_moment = torch.empty(3)

for image_path in images_path:
    x = Image.open(image_path).convert("RGB")
    x = TF.to_tensor(x)

    c, h, w = x.shape
    nb_pixels = h * w
    sum_ = torch.sum(x, dim=[1, 2])
    sum_of_square = torch.sum(x ** 2,
                              dim=[1, 2])
    fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
    snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
    cnt += nb_pixels

mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
print(mean, std)