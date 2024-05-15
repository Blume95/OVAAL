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
    parser = argparse.ArgumentParser(description='OVAAL')
    parser.add_argument('--root_path', type=str, default=pathlib.Path(__file__).parent.resolve().as_posix())
    parser.add_argument('--dataset', type=str, default='cv', choices=['cs', 'cv', 'kitti'])
    parser.add_argument('--dataset_path', type=str, default='/home/jing/Downloads/my_data/cv')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--pixels_per_image_per_class_pre_round', type=float, default=10.0)
    parser.add_argument('--initial_round', type=str, default='random', choices=['random', 'human'])
    parser.add_argument('--selection_round', type=int, default=5)
    parser.add_argument('--share_info', type=bool, default=False)
    parser.add_argument('--selection_strategy', type=str, default='entropy', choices=["least_confidence", "entropy",
                                                                                      "random", "pred_edges",
                                                                                      "edge_entropy"])
    parser.add_argument('--heatmap', type=bool, default=False)
    parser.add_argument('--pixelpick', type=bool, default=False)
    # parser.add_argument('--ensemble_type', type=str, default='WTA', choices=['WTA', ''])
    parser.add_argument('--fully_supervised', type=bool, default=False)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--ignore_index', type=int, default=255)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--lr_scheduler_type", type=str, default="Poly", choices=["MultiStepLR", "Poly"])

    args = parser.parse_args()

    index_label = used_classes(args.dataset)
    if args.fully_supervised:
        exp_name = f"{args.dataset}_{args.epochs}_fully_supervised"
    else:
        exp_name = f"{args.dataset}_{args.epochs}_{args.selection_strategy}_{args.pixels_per_image_per_class_pre_round}_{args.share_info}"
    st_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.root_path, f"output/train_{exp_name}")
    initial_selection_file = os.path.join(args.root_path,
                                          f"output/initial_selection/{args.dataset}_{args.initial_round}"
                                          f"_{int(len(index_label) * args.pixels_per_image_per_class_pre_round)}.pkl")

    args.save_dir = save_dir
    args.st_time = st_time
    args.index_label = index_label
    args.initial_selection_file = initial_selection_file

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.root_path, "output/initial_selection"), exist_ok=True)

    with open(os.path.join(save_dir, 'config.yaml'), 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    return args


def _train_epoch(epoch, model, dataloader_train, optimizer, lr_scheduler, label_name, device, args,
                 running_score, logger, uncertainty_sampler, model_support=None):
    model.train()
    dataloader_iter, tbar = iter(dataloader_train), tqdm(range(len(dataloader_train)))

    if uncertainty_sampler is None:
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
            if args.heatmap:
                prob = nn.Sigmoid()(logits)
                pred = (prob > 0.5).int()
                loss_cross_entropy = nn.MSELoss(reduction='none')(prob[:, 0, :, :], y.to(torch.float))
                loss_cross_entropy = (loss_cross_entropy * (y == args.ignore_index)).mean()

            else:
                prob, pred = F.softmax(logits.detach(), dim=1), logits.argmax(dim=1)
                # print(pred.shape)
                # print(torch.sum(pred))
                # print((pred.shape[0]*pred.shape[1]*pred.shape[2] - torch.sum(pred)) / torch.sum(pred))

                # weights = torch.tensor([1, (pred.shape[0]*pred.shape[1]*pred.shape[2] - torch.sum(pred)) / torch.sum(pred)], dtype=torch.float).to(device)

                loss_cross_entropy = F.cross_entropy(logits, y, ignore_index=args.ignore_index)

            optimizer.zero_grad()
            loss_cross_entropy.backward()
            optimizer.step()

            running_score.update(y.cpu().numpy(), pred.cpu().numpy())
            scores = running_score.get_scores()
    else:
        for _ in tbar:
            dict_data = next(dataloader_iter)
            x, y = dict_data["x"].to(device), dict_data["y"].to(device)

            dict_outputs = model(x)
            dict_outputs_support = model_support(x)
            logits_support = dict_outputs_support["pred"]
            logits = dict_outputs["pred"]

            prob, pred = F.softmax(logits.detach(), dim=1), logits.argmax(dim=1)
            prob_support, pred_support = F.softmax(logits_support.detach(), dim=1), logits_support.argmax(dim=1)
            unc_map = uncertainty_sampler(prob_support)
            unc_map_flatten = unc_map.flatten()

            topk_value = torch.max(unc_map_flatten.topk(k=1000, dim=0, largest=True).values)
            pred_support[unc_map > topk_value] = ~pred_support[unc_map > topk_value]

            loss_cross_entropy = F.cross_entropy(logits, pred_support, ignore_index=args.ignore_index)

            optimizer.zero_grad()
            loss_cross_entropy.backward()
            optimizer.step()

            running_score.update(y.cpu().numpy(), pred.cpu().numpy())
            scores = running_score.get_scores()

    lr_scheduler.step(epoch=epoch - 1)

    description = (f"Training_class: {label_name} Epoch : {epoch} | Foreground IoU : {scores['IoU'][1]:.4f} "
                   f"Background IoU : {scores['IoU'][0]:.4f} | Loss : {loss_cross_entropy} | ")
    tbar.set_description(description)
    logger.info(tbar)
    running_score.reset()
    return model, optimizer, lr_scheduler


def _val(model, epoch, dataloader_val, training_class_nth_dir, best_epoch_weight_name,
         final_epoch_weight_name, device, running_score, label_name, logger, args, best_iou=-1):
    model.eval()
    dataloader_iter, tbar = iter(dataloader_val), tqdm(range(len(dataloader_val)))
    with torch.no_grad():
        for _ in tbar:
            dict_data = next(dataloader_iter)
            # y b x 1 x w x h
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            dict_outputs = model(x)
            logits = dict_outputs['pred']
            if args.heatmap:
                prob = nn.Sigmoid()(logits)
                pred = (prob > 0.5).int()
            else:
                pred = logits.argmax(dim=1)

            running_score.update(y.cpu().numpy(), pred.cpu().numpy())

    scores = running_score.get_scores()
    description = (f"Training_class: {label_name} Epoch : {epoch} | Foreground IoU : {scores['IoU'][1]:.4f} "
                   f"Background IoU : {scores['IoU'][0]:.4f} | ")
    tbar.set_description(description)
    logger.info(tbar)
    running_score.reset()

    if epoch == args.epochs - 1:
        state_dict = {"model": model.state_dict()}
        torch.save(state_dict, f"{training_class_nth_dir}/{final_epoch_weight_name}")
        save_pkl(scores, f"{training_class_nth_dir}/final_epoch_evaluation_metrics.pkl")
        running_score.reset()
    if scores['IoU'][0] > best_iou:
        state_dict = {"model": model.state_dict()}
        torch.save(state_dict, f"{training_class_nth_dir}/{best_epoch_weight_name}")
        running_score.reset()
        return scores['IoU'][0]

    return -1


def main():
    args = parse_config()
    log_file = os.path.join(args.save_dir, f"train_{args.st_time}.log")
    logger = create_logger(log_file)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    running_score = RunningScore(2)
    best_epoch_weight_name = "best_epoch_model.pt"
    final_epoch_weight_name = "final_epoch_model.pt"

    logger.info('********************** Start logging **********************')
    index_label = used_classes(args.dataset)
    logger.info(f"{args}")
    logger.info(f"{index_label}")
    best_iou_dict = {x: -1 for x in index_label.keys()}
    uncertainty_sampler = UncertaintySampler(args.selection_strategy)

    if args.fully_supervised:
        for index, label_name in index_label.items():
            save_dir_path = args.save_dir + f"/{label_name}"
            os.makedirs(save_dir_path, exist_ok=True)
            logger.info("----------- Create dataloader & network & optimizer -----------")
            datasets_train = define_dataset(args.dataset, 'train', args.dataset_path, index, fully_supervised=True)
            datasets_val = define_dataset(args.dataset, 'val', args.dataset_path, index, fully_supervised=True)

            args.mean, args.std = datasets_train.mean, datasets_train.std

            dataloader_val = get_dataloader(datasets_val, 6, args.n_workers, shuffle_=False)
            dataloader_train = get_dataloader(datasets_train, args.batch_size, args.n_workers, shuffle_=True)
            if args.heatmap:
                model = Deeplab(class_num=1).to(device)
            else:
                model = Deeplab(class_num=2).to(device)
            optimizer = get_optimizer(model, case_=args.dataset)

            lr_scheduler = get_lr_scheduler(args.epochs, optimizer, len(dataloader_train),
                                            lr_scheduler_type=args.lr_scheduler_type)
            logger.info("----------- Start training  -----------")

            for e in range(args.epochs):
                model, optimizer, lr_scheduler = _train_epoch(e, model, dataloader_train, optimizer, lr_scheduler,
                                                              label_name, device, args, running_score, logger, None)
                logger.info(f"----------- {e} epochs validation  -----------")
                score = _val(model, e, dataloader_val, save_dir_path, best_epoch_weight_name, final_epoch_weight_name,
                             device, running_score, label_name, logger, args, best_iou=best_iou_dict[index])
                if score != -1:
                    best_iou_dict[index] = score
    else:
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

            if args.share_info or selection_round == 0:
                query_dict = read_pkl(os.path.join(nth_folder_path, "queries.pkl"))
                query_dict_tmp = query_dict

            if os.path.exists(args.save_dir + f"/{selection_round + 1}_th/queries.pkl"):
                logger.info(f'********************** Skip selection round {selection_round}, because this round has been '
                            f'trained **********************')
                continue

            for index, label_name in index_label.items():
                logger.info(
                    f'********************** Creating {label_name} of {selection_round} round **********************')
                category_of_nth = nth_folder_path + f"/{label_name}"
                os.makedirs(category_of_nth, exist_ok=True)

                last_round_best_weights = find_best_weights(args.save_dir, label_name)
                this_round_final_weights = args.save_dir + f"/{selection_round}_th/{label_name}/{final_epoch_weight_name}"
                if os.path.exists(this_round_final_weights):
                    logger.info(
                        f'********************** Skip Category {label_name}, because this Category has been '
                        f'trained **********************')
                    continue

                if args.share_info is False and selection_round > 0:
                    query_dict = read_pkl(nth_folder_path + f"/{label_name}_queries.pkl")

                logger.info("----------- Create dataloader & network & optimizer -----------")
                datasets_train = define_dataset(args.dataset, 'train', args.dataset_path, index)
                datasets_val = define_dataset(args.dataset, 'val', args.dataset_path, index)

                datasets_train.update_queries(query_dict)
                args.mean, args.std = datasets_train.mean, datasets_train.std

                dataloader_val = get_dataloader(datasets_val, 6, args.n_workers, shuffle_=False)
                dataloader_train = get_dataloader(datasets_train, args.batch_size, args.n_workers, shuffle_=True)

                model = Deeplab(class_num=2).to(device)
                if last_round_best_weights is not None:
                    state_dict: dict = torch.load(last_round_best_weights, map_location=device)["model"]
                    model.load_state_dict(state_dict)
                    logger.info(f"----------- load {last_round_best_weights} to GPU -----------")

                optimizer = get_optimizer(model, case_=args.dataset)

                lr_scheduler = get_lr_scheduler(args.epochs, optimizer, len(dataloader_train),
                                                lr_scheduler_type=args.lr_scheduler_type)
                logger.info("----------- Start training  -----------")
                for e in range(args.epochs):
                    model, optimizer, lr_scheduler = _train_epoch(e, model, dataloader_train, optimizer, lr_scheduler,
                                                                  label_name, device, args, running_score, logger,
                                                                  uncertainty_sampler=None)
                    logger.info(f"----------- {e} epochs validation  -----------")
                    score = _val(model, e, dataloader_val, category_of_nth, best_epoch_weight_name, final_epoch_weight_name,
                                 device, running_score, label_name, logger, args, best_iou=best_iou_dict[index])
                    if score != -1:
                        best_iou_dict[index] = score

                # do active selection
                if args.share_info:
                    query_dict_tmp = active_selection(args, query_dict_tmp, model, index, device)
                else:
                    query_dict_tmp = active_selection(args, query_dict, model, index, device)
                    query_path = args.save_dir + f"/{selection_round + 1}_th/{label_name}_queries.pkl"
                    os.makedirs(args.save_dir + f"/{selection_round + 1}_th", exist_ok=True)
                    if not os.path.exists(query_path):
                        save_pkl(query_dict_tmp, query_path)

            if args.share_info:
                query_path = args.save_dir + f"/{selection_round + 1}_th/queries.pkl"
                os.makedirs(args.save_dir + f"/{selection_round + 1}_th", exist_ok=True)
                if not os.path.exists(query_path):
                    save_pkl(query_dict_tmp, query_path)

    for index, label_name in index_label.items():
        category_of_semi = args.save_dir + f"/semi/{label_name}"
        os.makedirs(category_of_semi, exist_ok=True)
        last_round_best_weights = find_best_weights(args.save_dir, label_name)

        logger.info("----------- Create dataloader & network & optimizer -----------")
        datasets_train = define_dataset(args.dataset, 'train', args.dataset_path, index)
        datasets_val = define_dataset(args.dataset, 'val', args.dataset_path, index)

        datasets_train.update_queries(query_dict)
        args.mean, args.std = datasets_train.mean, datasets_train.std

        dataloader_val = get_dataloader(datasets_val, 6, args.n_workers, shuffle_=False)
        dataloader_train = get_dataloader(datasets_train, int(args.batch_size / 4), args.n_workers, shuffle_=True)

        model = Deeplab(class_num=2).to(device)
        model_support = Deeplab(class_num=2).to(device)
        if last_round_best_weights is not None:
            state_dict: dict = torch.load(last_round_best_weights, map_location=device)["model"]
            model.load_state_dict(state_dict)
            model_support.load_state_dict(state_dict)
            logger.info(f"----------- load {last_round_best_weights} to GPU -----------")

        optimizer = get_optimizer(model, case_=args.dataset)

        lr_scheduler = get_lr_scheduler(args.epochs, optimizer, len(dataloader_train),
                                        lr_scheduler_type=args.lr_scheduler_type)
        logger.info("----------- Start training  -----------")
        for e in range(args.epochs):
            model, optimizer, lr_scheduler = _train_epoch(e, model, dataloader_train, optimizer, lr_scheduler,
                                                          label_name, device, args, running_score, logger,
                                                          uncertainty_sampler=uncertainty_sampler,
                                                          model_support=model_support)
            logger.info(f"----------- {e} epochs validation  -----------")
            score = _val(model, e, dataloader_val, category_of_semi, best_epoch_weight_name, final_epoch_weight_name,
                         device, running_score, label_name, logger, args, best_iou=best_iou_dict[index])
            if score != -1:
                best_iou_dict[index] = score


if __name__ == "__main__":
    main()
