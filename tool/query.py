import numpy as np
from PIL import Image
from tool.utils import used_classes, save_pkl
from datasets.dataset_template import get_label_image_path
import torch
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2


def encode_query(name, size, query):
    y_coords, x_coords = np.where(query)
    query_info = {
        name: {
            "height": size[0],
            "width": size[1],
            "x_coords": x_coords,
            "y_coords": y_coords
        }
    }
    return query_info


def num_query_pixels(query_dict):
    num_pixels = 0
    for p_img, info_dict in query_dict.items():
        num_pixels += len(info_dict['x_coords'])
    return num_pixels


def query_decoder(query_info):
    query: np.ndarray = np.zeros((query_info["height"], query_info["width"]), dtype=bool)
    queried_pixels = zip(query_info["y_coords"], query_info["x_coords"])
    for i, loc in enumerate(queried_pixels):
        query[loc] = True
    return query


def process_x(p_img, device, args):
    x = Image.open(p_img).convert("RGB")
    x = TF.normalize(TF.to_tensor(x), args.mean, args.std)
    x = x[None, :, :, :]
    x = x.to(device)
    return x


class UncertaintySampler:
    def __init__(self, query_strategy):
        self.query_strategy = query_strategy

    @staticmethod
    def _entropy(prob):
        # 0-0.6931471805599453
        return (-prob * torch.log(prob)).sum(dim=1)  # b x h x w

    @staticmethod
    def _least_confidence(prob):
        # 0-0.4999
        return 1.0 - prob.max(dim=1)[0]  # b x h x w

    @staticmethod
    def _margin_sampling(prob):
        # 0-1
        top2 = prob.topk(k=2, dim=1).values  # b x k x h x w
        return (top2[:, 0, :, :] - top2[:, 1, :, :]).abs()  # b x h x w

    @staticmethod
    def _random(prob):
        b, _, h, w = prob.shape
        return torch.rand((b, h, w))

    @staticmethod
    def _pred_edges(prob):
        # input_prob 1, 2, 256, 512
        prob = prob[0, 1, :, :]  # 256 512
        prob_numpy = prob.numpy()
        sobelx = cv2.Sobel(prob_numpy, cv2.CV_64F, 1, 0, ksize=3)  # x
        sobely = cv2.Sobel(prob_numpy, cv2.CV_64F, 0, 1, ksize=3)  # y

        xy = np.sqrt(sobelx ** 2 + sobely ** 2)
        xy = xy[None, :, :]
        edges_tensor = torch.from_numpy(xy)
        return edges_tensor

    def _edge_entropy(self, prob):
        edges_tensor = self._pred_edges(prob)
        entropy_tensor = self._entropy(prob)
        edges_tensor = edges_tensor / (torch.max(edges_tensor) + 1e-9)
        entropy_tensor = entropy_tensor / (torch.max(entropy_tensor) + 1e-9)

        unc = edges_tensor + entropy_tensor
        return unc

    def __call__(self, prob):
        return getattr(self, f"_{self.query_strategy}")(prob)


def random_selection(args):
    query_dict = dict()
    index_label = used_classes(args.dataset)
    if args.pixelpick:
        pixel_per_image = args.pixels_per_image_per_class_pre_round
    else:
        pixel_per_image = args.pixels_per_image_per_class_pre_round * len(index_label)
    int_part = int(pixel_per_image)
    float_part = pixel_per_image - int_part
    _, label_path_list = get_label_image_path(args.dataset_path, 'train', None, None)

    selected_labels = np.random.choice(label_path_list, int(float_part * len(label_path_list)), replace=False)

    for label_path in label_path_list:
        label_data = np.array(Image.open(label_path))
        interested_mask = label_data != args.ignore_index
        interested_pixels = np.where(interested_mask.flatten())[0]

        query_flat = np.zeros((label_data.shape[0] * label_data.shape[1]), dtype=bool)
        if label_path in selected_labels:
            selected_pixels = np.random.choice(interested_pixels, int_part + 1, replace=False)
        else:
            selected_pixels = np.random.choice(interested_pixels, int_part, replace=False)

        query_flat[selected_pixels] = True

        queries = query_flat.reshape((label_data.shape[0], label_data.shape[1]))
        query_info = encode_query(label_path.split("/")[-1], (label_data.shape[0], label_data.shape[1]),
                                  queries)
        query_dict.update(query_info)

    save_pkl(query_dict, args.initial_selection_file)


def active_selection(args, query_dict, model, used_class, device):
    model.eval()
    prev_pixels_num = num_query_pixels(query_dict)
    uncertainty_sampler = UncertaintySampler(args.selection_strategy)

    index_label = used_classes(args.dataset)
    pixel_per_image = args.pixels_per_image_per_class_pre_round
    int_part = int(pixel_per_image)
    float_part = pixel_per_image - int_part
    _, label_path_list = get_label_image_path(args.dataset_path, 'train', index_label, used_class)
    selected_labels = np.random.choice(label_path_list, int(float_part * len(label_path_list)), replace=False)

    with torch.no_grad():
        for name, query_info in tqdm(query_dict.items()):
            h, w = query_info["height"], query_info["width"]
            x = process_x(args.dataset_path + f"/images/{name}", device, args)
            dict_outputs = model(x)
            logist = dict_outputs['pred'].cpu()
            input_prob = F.softmax(logist, dim=1)
            queried_pixels = list(zip(query_info["y_coords"], query_info["x_coords"]))
            unc_map = uncertainty_sampler(input_prob)[0]  # hx w
            label = np.array(Image.open(args.dataset_path + f"/labels/{name}"))
            ignore_area = label == args.ignore_index
            unc_map[ignore_area] = 0.

            for i, loc in enumerate(queried_pixels):
                unc_map[loc] = 0.

            unc_map = unc_map.flatten()

            if args.dataset_path + f"/labels/{name}" in selected_labels:
                ind_queries = unc_map.topk(k=int_part + 1, dim=0, largest=True).indices.cpu().numpy()
            else:
                ind_queries = unc_map.topk(k=int_part, dim=0, largest=True).indices.cpu().numpy()

            query = np.zeros(h * w, dtype=bool)
            query[ind_queries] = True
            query = query.reshape((h, w))

            for loc in queried_pixels:
                query[loc] = True

            query_info_new = encode_query(name, (h, w), query)
            query_dict.update(query_info_new)
    now_pixels_num = num_query_pixels(query_dict)
    print(f"From previous {prev_pixels_num} pixels to now {now_pixels_num}")

    return query_dict
