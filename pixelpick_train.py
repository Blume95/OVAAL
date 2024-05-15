import os
import argparse
import pathlib
from datetime import datetime

import torch
import yaml
from tool.utils import *
from tool.query import *
import shutil
from networks.deeplab import Deeplab
from tqdm import tqdm
import torch.nn.functional as F
from time import time
import torch.nn as nn


def parse_config():
    parser = argparse.ArgumentParser(description='pixelpick')
    parser.add_argument('--root_path', type=str, default=pathlib.Path(__file__).parent.resolve().as_posix())
    parser.add_argument('--dataset', type=str, default='cv', choices=['cs', 'cv', 'kitti'])
    parser.add_argument('--dataset_path', type=str, default='/home/jing/Downloads/my_data/cv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--pixels_per_image_per_class_pre_round', type=float, default=1.0)
    parser.add_argument('--initial_round', type=str, default='random', choices=['random', 'human'])
    parser.add_argument('--selection_round', type=int, default=10)
    parser.add_argument('--selection_strategy', type=str, default='margin_sampling', choices=["least_confidence", "entropy",
                                                                                      "random", "pred_edges",
                                                                                      "edge_entropy"])
    parser.add_argument('--fully_supervised', type=bool, default=False)
    parser.add_argument('--pixelpick', type=bool, default=True)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignore_index', type=int, default=255)
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument("--lr_scheduler_type", type=str, default="Poly", choices=["MultiStepLR", "Poly"])

    args = parser.parse_args()
    index_label = used_classes(args.dataset)
    args.pixels_per_image_per_class_pre_round = args.pixels_per_image_per_class_pre_round * len(index_label)
    if args.fully_supervised:
        exp_name = f"pixelpick_{args.dataset}_{args.epochs}_fully_supervised"
    else:
        exp_name = f"pixelpick_{args.dataset}_{args.epochs}_{args.selection_strategy}_{args.pixels_per_image_per_class_pre_round}"
    st_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.root_path, f"output/train_{exp_name}")
    initial_selection_file = os.path.join(args.root_path,
                                          f"output/initial_selection/pixelpick_{args.dataset}_{args.initial_round}"
                                          f"_{args.pixels_per_image_per_class_pre_round}.pkl")

    args.save_dir = save_dir
    args.st_time = st_time
    args.index_label = index_label
    args.initial_selection_file = initial_selection_file

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.root_path, "output/initial_selection"), exist_ok=True)

    with open(os.path.join(save_dir, 'config.yaml'), 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    return args


def _train_epoch(epoch, model, dataloader_train, optimizer, lr_scheduler, device, args,
                 running_score, logger):
    model.train()
    dataloader_iter, tbar = iter(dataloader_train), tqdm(range(len(dataloader_train)))

    for _ in tbar:
        dict_data = next(dataloader_iter)
        # y: b x w x h
        # mask: b x w x h
        # x: b x 3 x w x h
        x, y = dict_data["x"].to(device), dict_data["y"].to(device)
        if not args.fully_supervised:
            mask = dict_data["queries"].to(device)
            y.flatten()[~mask.flatten()] = args.ignore_index

        dict_outputs = model(x)
        # batch_size, 1, h,w
        logits = dict_outputs["pred"]
        prob, pred = F.softmax(logits.detach(), dim=1), logits.argmax(dim=1)

        # weights = torch.tensor([0.5, 2], dtype=torch.float).to(device)

        loss_cross_entropy = F.cross_entropy(logits, y, ignore_index=args.ignore_index)

        optimizer.zero_grad()
        loss_cross_entropy.backward()
        optimizer.step()

        running_score.update(y.cpu().numpy(), pred.cpu().numpy())
        scores = running_score.get_scores()

    lr_scheduler.step(epoch=epoch - 1)

    description = (f"Epoch : {epoch} | IoU : {scores['IoU']} | Loss : {loss_cross_entropy} | ")
    tbar.set_description(description)
    logger.info(tbar)
    running_score.reset()
    return model, optimizer, lr_scheduler


def _val(model, epoch, dataloader_val, training_class_nth_dir,
         final_epoch_weight_name, device, running_score, logger, best_iou):
    model.eval()
    dataloader_iter, tbar = iter(dataloader_val), tqdm(range(len(dataloader_val)))
    with torch.no_grad():
        for _ in tbar:
            dict_data = next(dataloader_iter)
            # y b x 1 x w x h
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            dict_outputs = model(x)
            logits = dict_outputs['pred']
            pred = logits.argmax(dim=1)

            running_score.update(y.cpu().numpy(), pred.cpu().numpy())

    scores = running_score.get_scores()
    description = (f"Epoch : {epoch} | mean IoU : {scores['Mean IoU']} | ")
    tbar.set_description(description)
    logger.info(tbar)

    if scores['Mean IoU'] > best_iou:
        state_dict = {"model": model.state_dict()}
        torch.save(state_dict, f"{training_class_nth_dir}/{final_epoch_weight_name}")
        save_pkl(scores, f"{training_class_nth_dir}/final_epoch_evaluation_metrics.pkl")
    running_score.reset()
    return best_iou


def main():
    args = parse_config()
    log_file = os.path.join(args.save_dir, f"train_{args.st_time}.log")
    logger = create_logger(log_file)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    running_score = RunningScore(len(args.index_label))
    final_epoch_weight_name = "final_epoch_model.pt"

    logger.info('********************** Start logging **********************')
    index_label = used_classes(args.dataset)
    logger.info(f"{args}")
    logger.info(f"{index_label}")

    for selection_round in range(args.selection_round):
        nth_folder_path = args.save_dir + f"/{selection_round}_th"
        if selection_round == 0:
            # check the initial selection
            logger.info(f'********************** Creating the {selection_round}_th folder **********************')
            os.makedirs(nth_folder_path, exist_ok=True)
            if os.path.exists(args.initial_selection_file):
                pass
            else:
                logger.info('********************** Doing Initial Selection **********************')
                random_selection(args)
            logger.info('********************** Finish Initial Selection **********************')
            shutil.copy(args.initial_selection_file, os.path.join(nth_folder_path, "queries.pkl"))

        query_dict = read_pkl(os.path.join(nth_folder_path, "queries.pkl"))
        now_pixels_num = num_query_pixels(query_dict)
        logger.info(f"----------- Using {now_pixels_num} Pixels for training  -----------")
        logger.info("----------- Create dataloader & network & optimizer -----------")
        datasets_train = define_dataset(args.dataset, 'train', args.dataset_path, used_class=None)
        datasets_val = define_dataset(args.dataset, 'val', args.dataset_path, used_class=None)

        datasets_train.update_queries(query_dict)
        args.mean, args.std = datasets_train.mean, datasets_train.std

        dataloader_val = get_dataloader(datasets_val, 6, args.n_workers, shuffle_=False)
        dataloader_train = get_dataloader(datasets_train, args.batch_size, args.n_workers, shuffle_=True)

        model = Deeplab(class_num=len(args.index_label)).to(device)

        optimizer = get_optimizer(model, case_=args.dataset)

        lr_scheduler = get_lr_scheduler(args.epochs, optimizer, len(dataloader_train),
                                        lr_scheduler_type=args.lr_scheduler_type)
        logger.info("----------- Start training  -----------")
        best_iou = -1

        for e in range(args.epochs):
            model, optimizer, lr_scheduler = _train_epoch(e, model, dataloader_train, optimizer, lr_scheduler, device, args, running_score, logger)
            logger.info(f"----------- {e} epochs validation  -----------")
            best_iou = _val(model, e, dataloader_val, nth_folder_path, final_epoch_weight_name,
                         device, running_score, logger, best_iou)

        query_dict = active_selection(args, query_dict, model, None, device)

        query_path = args.save_dir + f"/{selection_round + 1}_th/queries.pkl"
        os.makedirs(args.save_dir + f"/{selection_round + 1}_th", exist_ok=True)
        if not os.path.exists(query_path):
            save_pkl(query_dict, query_path)


if __name__ == "__main__":
    main()
