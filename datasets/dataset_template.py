import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from random import random, randint, uniform
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode as IM
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale
import cv2
import os


class DatasetTemplate(Dataset):
    def __init__(self):
        super(DatasetTemplate, self).__init__()
        self.query_dict = None
        self.use_aug = True
        self.augmentations = {
            "random_scale": self.use_aug,
            "random_hflip": self.use_aug,
            "crop": self.use_aug,
            "random_color_jitter": self.use_aug,
            "random_grayscale": self.use_aug,
            "random_gaussian_blur": self.use_aug
        }

    def update_queries(self, query_dict):
        self.query_dict = query_dict

    @staticmethod
    def query_decoder(query_info):
        query: np.ndarray = np.zeros((query_info["height"], query_info["width"]), dtype=bool)
        queried_pixels = zip(query_info["y_coords"], query_info["x_coords"])
        for i, loc in enumerate(queried_pixels):
            query[loc] = True
        return query

    def _augmentations(self, x: Image.Image, y: Image.Image, query: torch.Tensor):
        if self.augmentations["random_scale"]:
            w, h = x.size
            random_scale = uniform(0.5, 2.0)
            w_rs, h_rs = int(w * random_scale), int(h * random_scale)
            x = TF.resize(x, (h_rs, w_rs), IM.BILINEAR)
            if y is not None:
                y = TF.resize(y, (h_rs, w_rs), IM.NEAREST)
            if query is not None:
                query = TF.resize(query.unsqueeze(0), (h_rs, w_rs), IM.NEAREST).squeeze(0)

        if self.augmentations["crop"]:
            # After random scale the size of image changed
            w, h = x.size
            pad_h, pad_w = max(self.crop_size[0] - h, 0), max(self.crop_size[1] - w, 0)
            self.pad_size = (pad_h, pad_w)
            # left top right bottom
            x = TF.pad(x, (0, 0, pad_w, pad_h), fill=self.mean_val, padding_mode="constant")
            if y is not None:
                y = TF.pad(y, (0, 0, pad_w, pad_h), fill=self.ignore_index, padding_mode="constant")
            if query is not None:
                query = TF.pad(query, (0, 0, pad_w, pad_h), fill=False, padding_mode="constant")

            w, h = x.size
            start_h, start_w = randint(0, h - self.crop_size[0]), randint(0, w - self.crop_size[1])

            x = TF.crop(x, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])
            if y is not None:
                y = TF.crop(y, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])
            if query is not None:
                query = TF.crop(query, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])

        if self.augmentations["random_hflip"]:
            if random() > 0.5:
                x = TF.hflip(x)
                if y is not None:
                    y = TF.hflip(y)
                if query is not None:
                    query = TF.hflip(query)

        if self.augmentations["random_color_jitter"]:
            color_jitter = ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            x = RandomApply([color_jitter], p=0.8)(x)
        if self.augmentations["random_grayscale"]:
            x = RandomGrayscale(0.2)(x)
        if self.augmentations["random_gaussian_blur"]:
            w, h = x.size
            smaller_length = min(w, h)
            x = GaussianBlur(kernel_size=int((0.1 * smaller_length // 2 * 2) + 1))(x)

        return x, y, query

    def __len__(self):
        return len(self.list_inputs)

    def __getitem__(self, item):
        # the same index in label and image list , should be the same file
        dict_data = dict()
        p_img = self.list_inputs[item]
        p_label = self.list_labels[item]

        image_name = p_img.split("/")[-1]

        x = Image.open(p_img).convert("RGB")
        y = Image.open(p_label)

        if self.mode == "train":
            if self.fully_supervised:
                x, y, query_tensor = self._augmentations(x, y, None)
                query_tensor = torch.zeros(1, 1)
            else:
                query_info = self.query_dict[image_name]
                query = self.query_decoder(query_info)
                query_tensor = torch.tensor(query)
                x, y, query_tensor = self._augmentations(x, y, query_tensor)

            dict_data.update({
                "x": TF.normalize(TF.to_tensor(x), self.mean, self.std),
                "y": torch.tensor(np.asarray(y, np.int64), dtype=torch.long),
                "p_img": p_img,
                "queries": query_tensor
            })

        elif self.mode == "val":
            dict_data.update({
                "x": TF.normalize(TF.to_tensor(x), self.mean, self.std),
                "y": torch.tensor(np.asarray(y, np.int64), dtype=torch.long),
                "p_img": p_img
            })
        else:
            raise NotImplementedError

        return dict_data


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


def get_label_image_path(dataset_path, mode, index_label, used_class):
    with open(os.path.join(dataset_path, f"ImageSets/{mode}.txt"), 'r') as f:
        names = f.readlines()
        names = [x.strip("\n") for x in names]
    list_inputs = [os.path.join(dataset_path, f"images/{x}") for x in names]
    if used_class is None:
        list_labels = [os.path.join(dataset_path, f"labels/{x}") for x in names]
    elif used_class is not None and index_label is not None:
        list_labels = [os.path.join(dataset_path, f"{index_label[used_class]}_labels/{x}") for x in names]
    return list_inputs, list_labels
