from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import cv2
from PyQt5 import QtGui
import os
import pickle as pkl
import numpy as np
from PIL import Image
from glob import glob
import numpy as np
# from training import Model
from networks.deeplab import Deeplab
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from query import UncertaintySampler
from datasets.human_annotation import ManualDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler


def save_pkl(pkl_file, pkl_name):
    print(pkl_name)
    with open(pkl_name, "wb") as f:
        pkl.dump(pkl_file, f)


def read_pkl(pkl_name):
    with open(pkl_name, "rb") as f:
        data = pkl.load(f)
    return data


def get_dataloader(dataset, batch_size, num_workers, shuffle_):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle_,
        drop_last=len(dataset) % batch_size == 1
    )

    return dataloader


def add_list_widget(dict_used_classes):
    list_widget = QListWidget()
    for k, v in dict_used_classes.items():
        color_square = np.ones((30, 30, 3), dtype=np.uint8)
        color_square[:, :, :] = palette_cs[k]

        item = QListWidgetItem(str(k) + f" : {v}")
        item.setIcon(QIcon(QPixmap.fromImage(cv_to_qt(color_square))))
        list_widget.addItem(item)
    return list_widget


def make_an_action(action_name, action_tips, icon_path,
                   check=False, enable=True):
    my_action = QAction()
    my_action.setText(action_name)
    my_action.setToolTip(action_tips)
    my_action.setCheckable(check)
    my_action.setEnabled(enable)
    my_action.setIcon(QIcon(icon_path))
    return my_action


def cv_to_qt(cv_img):
    temp_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    qt_img = QtGui.QImage(temp_img.data,
                          temp_img.shape[1],
                          temp_img.shape[0],
                          temp_img.shape[1] * 3,
                          QtGui.QImage.Format_RGB888)
    return qt_img


class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor = pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        assert factor >= 0, 'error in lr_scheduler'
        return [base_lr * factor for base_lr in self.base_lrs]


def get_optimizer(model, case_="cityscapes"):
    if case_ == "cityscapes":
        optimizer_params = {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 2e-4,
            "eps": 1e-7
        }
        list_params = [{'params': model.backbone.parameters(),
                        'lr': optimizer_params['lr'] / 10,
                        'weight_decay': optimizer_params['weight_decay']}]

        list_params += [{'params': model.aspp.parameters(),
                         'lr': optimizer_params['lr'],
                         'weight_decay': optimizer_params['weight_decay']}]

        list_params += [{'params': model.low_level_conv.parameters(),
                         'lr': optimizer_params['lr'],
                         'weight_decay': optimizer_params['weight_decay']}]

        list_params += [{'params': model.seg_head.parameters(),
                         'lr': optimizer_params['lr'],
                         'weight_decay': optimizer_params['weight_decay']}]

        optimizer = Adam(list_params)

    return optimizer


def get_lr_scheduler(n_epochs, optimizer, iters_per_epoch=-1, lr_scheduler_type="Poly"):
    if lr_scheduler_type == "MultiStepLR":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    elif lr_scheduler_type == "Poly":
        lr_scheduler = Poly(optimizer, n_epochs, iters_per_epoch)
    return lr_scheduler


def start_training(bar, args, nth, device, batch_size=2, n_works=2):
    query_dict = read_pkl(f"{args.dir_root}/{args.project_name}/{nth}_query/queries.pkl")
    model_support = Deeplab(class_num=args.class_num, output_stride=args.stride_total,
                            pretrained=args.with_pretrain).to(device)
    model = Deeplab(class_num=args.class_num, output_stride=args.stride_total,
                    pretrained=args.with_pretrain).to(device)

    count = 0

    for index, used_class_name in args.used_classes_dict.items():
        # skip index =0 unknown
        if index == 0:
            continue
        dataset = ManualDataset(args, index)
        dataset.update_queries(query_dict)
        dataloader = get_dataloader(dataset, batch_size, n_works, True)
        total_num = len(args.class_names[1:]) * len(dataloader) * args.epochs
        support_weight_path = f"{args.support_weight_dir}/{used_class_name}_final_epoch_model.pt"
        prev_weight_path = f"{args.dir_root}/{args.project_name}/{nth - 1}_query/{used_class_name}_final_epoch_model.pt"

        optimizer = get_optimizer(model)
        lr_scheduler = get_lr_scheduler(args.epochs, optimizer, len(dataloader),
                                        lr_scheduler_type=args.lr_scheduler_type)

        if os.path.exists(support_weight_path):
            state_dict_support: dict = torch.load(support_weight_path, map_location=device)["model"]
            model_support.load_state_dict(state_dict_support)
        if os.path.exists(prev_weight_path):
            state_dict: dict = torch.load(prev_weight_path, map_location=device)["model"]
            model.load_state_dict(state_dict)

        for e in range(args.epochs):
            model.train()
            model_support.eval()
            dataloader_iter = iter(dataloader)
            for _ in range(len(dataloader)):
                count += 1
                dict_data = next(dataloader_iter)
                x, y = dict_data["x"].to(device), dict_data["y"].to(device)

                #
                dict_outputs = model(x)
                logits = dict_outputs["pred"]
                loss_gt = F.cross_entropy(logits, y, ignore_index=args.ignore_index)

                dict_outputs_support = model_support(x)
                logits_support = dict_outputs_support['pred']
                loss_sp = F.cross_entropy(logits, logits_support)

                if os.path.exists(support_weight_path):
                    loss_sum = loss_gt + loss_sp
                else:
                    loss_sum = loss_gt

                bar.setValue((count / total_num) * 100)

                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()

            lr_scheduler.step(epoch=e - 1)
        state_dict = {"model": model.state_dict()}
        torch.save(state_dict,
                   f"{args.dir_root}/{args.project_name}/{nth}_query/{used_class_name}_final_epoch_model.pt")


def prediction(bar, args, nth, device):
    model = Deeplab(class_num=args.class_num, output_stride=args.stride_total,
                    pretrained=args.with_pretrain).to(device)
    count = 0
    model.eval()
    dataset = ManualDataset(args, None, mode='vis')
    dataloader = get_dataloader(dataset, 1, 1, False)
    dataloader_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(len(dataloader)):
            dict_data = next(dataloader_iter)
            x = dict_data["x"].to(device)
            p_img = dict_data['p_img'][0]
            output = torch.zeros((x.shape[0], len(args.class_names), x.shape[2], x.shape[3]))
            output[:, 0:, :] = 0.5

            for i, used_class in enumerate(args.class_names[1:]):
                current_weight_path = f"{args.dir_root}/{args.project_name}/{nth}_query/{used_class}_final_epoch_model.pt"
                state_dict: dict = torch.load(current_weight_path, map_location=device)["model"]
                model.load_state_dict(state_dict)
                dict_outputs = model(x)
                logits = dict_outputs["pred"]
                prob = F.softmax(logits.detach(), dim=1)
                output[:, i + 1, :, :] = prob[:, 1, :, :]

            output_prob = torch.max(output, dim=1)
            output = output_prob.indices

            output = output[0]

            out = np.zeros((output.shape[0], output.shape[1], 3))

            for i in range(len(args.class_names)):
                out[output == i] = palette_cs[i]
            out = np.clip(out, 0, 255).astype(np.uint8)

            cv2.imwrite(f"{args.dir_root}/{args.project_name}/{nth}_query/mask_out/{p_img.split('/')[-1]}", out)


def encode_query_annotation(p_img, query, old_query_dict):
    y_coords, x_coords = np.where(query)
    query_info = {
        p_img: {(y_coords[i], x_coords[i]): -1 for i in range(len(x_coords))}
    }
    if p_img in old_query_dict:
        for k, v in old_query_dict[p_img].items():
            query_info[p_img][k] = v
    return query_info


def query_pixel_num(query_dict):
    num_pixels = 0
    for p_img, info_dict in query_dict.items():
        num_pixels += len(info_dict)
    return num_pixels


class Query:
    def __init__(self, args):
        self.args = args
        self.uncertainty_sampler = UncertaintySampler(self.args.selection_strategy)

    def process_x(self, p_img, device):
        x = Image.open(p_img).convert("RGB")
        x = TF.normalize(TF.to_tensor(x), self.args.mean, self.args.std)
        x = x[None, :, :, :]
        x = x.to(device)
        return x

    def random_selection(self, bar):
        query_dict = dict()
        count = 0
        total_num = len(self.args.class_names[1:]) * len(self.args.image_path_list)
        for used_class_names in self.args.class_names[1:]:
            float_part = self.args.budget_per_image_per_round_per_class - round(
                self.args.budget_per_image_per_round_per_class)
            selected_images = np.random.choice(self.args.image_path_list,
                                               int(float_part * len(self.args.image_path_list)),
                                               replace=False)
            for index, image_path in enumerate(self.args.image_path_list):
                count += 1
                int_part = round(self.args.budget_per_image_per_round_per_class)

                label_data = np.zeros_like(np.array(Image.open(image_path))[:, :, 0])
                w, h = label_data.shape[1], label_data.shape[0]

                if image_path in query_dict:
                    queried_pixels = list(query_dict[image_path].keys())
                else:
                    queried_pixels = []

                for loc in queried_pixels:
                    label_data[loc] = self.args.ignore_index

                interested_mask = label_data != self.args.ignore_index
                interested_pixels = np.where(interested_mask.flatten())[0]

                if image_path in selected_images:
                    int_part += 1

                queries_flat = np.zeros((h * w), dtype=bool)
                selected_pixels = np.random.choice(interested_pixels, int_part, replace=False)
                queries_flat[selected_pixels] = True

                queries = queries_flat.reshape((h, w))

                for loc in queried_pixels:
                    queries[loc] = True

                query_info = encode_query_annotation(image_path, queries, query_dict)
                query_dict.update(query_info)
                bar.setValue((count / total_num) * 100)

        save_pkl(query_dict, f"{self.args.dir_root}/{self.args.project_name}/0_query/queries.pkl")
        print(f"From previous 0 pixels to now {query_pixel_num(query_dict)}")

    def normal_round(self, nth, device, bar, model):
        count = 0
        total_num = len(self.args.class_names[1:]) * len(self.args.image_path_list)
        for used_class_name in self.args.class_names[1:]:

            now_pkl_path = f"{self.args.dir_root}/{self.args.project_name}/{nth}_query/queries.pkl"

            if os.path.exists(now_pkl_path):
                query_dict = read_pkl(now_pkl_path)
            else:
                if nth != 0:
                    prev_pkl_path = f"{self.args.dir_root}/{self.args.project_name}/{nth - 1}_query/queries.pkl"
                    query_dict = read_pkl(prev_pkl_path)
                else:
                    query_dict = dict()

            if nth != 0:
                prev_weight_path = f"{self.args.dir_root}/{self.args.project_name}/{nth - 1}_query/{used_class_name}_final_epoch_model.pt"
            else:
                prev_weight_path = f"{self.args.support_weight_dir}/{used_class_name}_final_epoch_model.pt"

            state_dict: dict = torch.load(prev_weight_path, map_location=device)["model"]
            model.load_state_dict(state_dict)
            model.eval()
            prev_pixels_num = query_pixel_num(query_dict)

            float_part = self.args.budget_per_image_per_round_per_class - round(
                self.args.budget_per_image_per_round_per_class)
            p_img_selected = np.random.choice(self.args.image_path_list,
                                              int(float_part * len(self.args.image_path_list)),
                                              replace=False)

            with torch.no_grad():
                queried_pixels = None
                for p_img in self.args.image_path_list:
                    int_part = round(self.args.budget_per_image_per_round_per_class)
                    x = self.process_x(p_img, device)
                    h, w = x.shape[2], x.shape[3]
                    uncertainty_range_num = int(self.args.uncertainty_ratio * h * w)
                    dict_outputs = model(x)
                    logist = dict_outputs['pred'].cpu()
                    input_prob = F.softmax(logist, dim=1)
                    unc_map = self.uncertainty_sampler(input_prob)[0]  # hx w

                    if p_img in query_dict:
                        queried_pixels = list(query_dict[p_img].keys())
                        for i, loc in enumerate(queried_pixels):
                            unc_map[loc] = 0.

                    unc_map = unc_map.flatten()

                    if p_img in p_img_selected:
                        int_part += 1
                    if uncertainty_range_num > 0:
                        ind_queries = unc_map.topk(k=uncertainty_range_num, dim=0,
                                                   largest=True).indices.cpu().numpy()
                        ind_queries = np.random.choice(ind_queries, int_part, False)
                    else:
                        ind_queries = unc_map.topk(k=int_part, dim=0, largest=True).indices.cpu().numpy()

                    query = np.zeros(h * w, dtype=bool)
                    query[ind_queries] = True
                    query = query.reshape((h, w))

                    if queried_pixels is not None:
                        for loc in queried_pixels:
                            query[loc] = True

                    query_info_new = encode_query_annotation(p_img, query, query_dict)
                    query_dict.update(query_info_new)
                    count += 1
                    bar.setValue((count / total_num) * 100)
                save_pkl(query_dict, now_pkl_path)
                now_pixels_num = query_pixel_num(query_dict)
                print(f"From previous {prev_pixels_num} pixels to now {now_pixels_num}")

    def __call__(self, nth, device, bar, model):
        support_weights_len = len(os.listdir(self.args.support_weight_dir))
        if nth == 0 and support_weights_len == 0:
            if self.args.initial_round == "human":
                1
                # todo: human selection

            elif self.args.initial_round == "random":
                self.random_selection(bar)
        else:
            self.normal_round(nth, device, bar, model)


palette_cs = {
    0: (0, 0, 0),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    19: (192, 125, 82),
    -1: (0, 0, 255)
}

if __name__ == "__main__":
    # unlabeled_image_dir = "/Annotation/Unlabeled"
    # image_list = os.listdir(unlabeled_image_dir)
    # img_path = [os.path.join(unlabeled_image_dir, image_name) for image_name in image_list]
    # initial_round(10, image_path_list=img_path,
    #               root_path="/mnt/Jan/Py_PJ/OVAAL_0.9/Annotation/Testing_annotation_0/testing_project/Selected")
    # prev_path = "/mnt/Jan/Py_PJ/OVAAL_0.9/Annotation/Testing_annotation_0/0_query/query.pkl"
    # now_path = "/mnt/Jan/Py_PJ/OVAAL_0.9/Annotation/Testing_annotation_0/1_query/query.pkl"
    # data_now = read_pkl(now_path)
    # data_pre = read_pkl(prev_path)
    # remove_repeat_xy(data_pre,data_now)
    # pre = "/mnt/Jan/Py_PJ/OVAAL_0.9/Annotation/Testing_annotation_0/0_query/annotated"
    # now = "/mnt/Jan/Py_PJ/OVAAL_0.9/Annotation/Testing_annotation_0/1_query/annotated"
    # merge_previous_pkl(pre, now)
    from time import time

    start = time()
    a = np.ones((1024, 2048), dtype=np.uint8) * 255
    a[8:9, :-1] = 0
    a[6:8, 0:3] = 1
    a[1:5, 4:9] = 2
    dict_test = {0: "wew", 1: "hhs", 2: "dsjkaj"}

    convert_array_to_pkl(a, dict_test)
