from tool.utils import *
from tqdm import tqdm
import torch.nn.functional as F
from networks.deeplab import Deeplab


def ensemble(index_path_dict: dict, dataloader, device):
    model = Deeplab(class_num=2).to(device)
    model.eval()
    running_score = RunningScore(len(index_path_dict))
    with torch.no_grad():
        dataloader_iter = iter(dataloader)
        for _ in tqdm(range(len(dataloader))):
            dict_data = next(dataloader_iter)
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            output = torch.zeros((x.shape[0], len(index_path_dict), x.shape[2], x.shape[3]))

            for i, (class_index, pt_path) in enumerate(index_path_dict.items()):
                # load data and model
                state_dict: dict = torch.load(pt_path, map_location=device)["model"]
                model.load_state_dict(state_dict)
                dict_outputs = model(x)
                logits = dict_outputs['pred']

                prob, pred = F.softmax(logits.detach(), dim=1), logits.argmax(dim=1)
                output[:, i, :, :] = prob[:, 1, :, :]

            output_prob = torch.max(output, dim=1)
            # background_mask = (output_prob.values < 0.5)
            output = output_prob.indices
            # output[background_mask] = len(index_path_dict)
            running_score.update(y.cpu().numpy(), output.cpu().numpy())
            scores = running_score.get_scores()
            print(scores['Mean IoU'])
            print(scores['IoU'])
        running_score.reset()


def _val(model, dataloader_val, device, class_num):
    model.eval()
    dataloader_iter, tbar = iter(dataloader_val), tqdm(range(len(dataloader_val)))
    running_score = RunningScore(class_num)
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
    print(scores['Mean IoU'])
    print(scores['IoU'])
    running_score.reset()


def main_pixepick(dataset_name, selection_round_num, device, root_path, dataset_path):
    index_label = used_classes(dataset_name)
    model = Deeplab(class_num=len(index_label)).to(device)
    model.eval()
    datasets_val = define_dataset(dataset_name, 'val', dataset_path, None)
    dataloader_val = get_dataloader(datasets_val, 8, 4, shuffle_=False)
    for i in range(selection_round_num):
        print(i)
        state_dict: dict = torch.load(f"{root_path}/{i}_th/final_epoch_model.pt", map_location=device)["model"]
        model.load_state_dict(state_dict)
        _val(model, dataloader_val, device, len(index_label))


def main(dataset_name, root_path, dataset_path, nth=-1):
    # evaluate Pixelpick and OVAAL
    index_label = used_classes(dataset_name)
    input_dict = dict()
    datasets_val = define_dataset(dataset_name, 'val', dataset_path, None)
    dataloader_val = get_dataloader(datasets_val, 6, 4, shuffle_=False)
    for index, label_name in index_label.items():
        if nth == -1:
            best_weight_path = find_best_weights(root_path, label_name)
            input_dict[index] = best_weight_path
        elif 'semi' in str(nth):
            input_dict[index] = f"{root_path}/{nth}/{label_name}/final_epoch_model.pt"
        else:
            input_dict[index] = f"{root_path}/{nth}_th/{label_name}/final_epoch_model.pt"
    print(input_dict)

    ensemble(input_dict, dataloader_val, "cuda:0")


if __name__ == "__main__":
    # main('cv', '/home/jing/Downloads/ICRA_2024/ICRA_NEW/output/train_cv_30_entropy_1.0_False',
    #      '/home/jing/Downloads/my_data/cv', nth='7_semi')
    # main('cv', '/home/jing/Downloads/ICRA_2024/ICRA_NEW/output/train_cv_30_entropy_1.0_False(backup)',
    #      '/home/jing/Downloads/my_data/cv', nth='semi')
    # main_pixepick('kitti', 10, "cuda:1",
    #               '/home/jing/Downloads/ICRA_2024/ICRA_NEW/tool/output/train_pixelpick_kitti_30_entropy_1.0',
    #               '/home/jing/Downloads/my_data/kitti_semantic')
    # main('kitti', '/home/jing/Downloads/ICRA_2024/ICRA_NEW/output/backup/train_kitti_30_entropy_1.0_False',
    #      '/home/jing/Downloads/my_data/kitti_semantic', nth=0)

    # main_pixepick('cv', 10, "cuda:1",
    #               '/home/jing/Downloads/ICRA_2024/ICRA_NEW/output/train_pixelpick_cv_50_entropy_11.0',
    #               '/home/jing/Downloads/my_data/cv')

    # main('cv', '/home/jing/Downloads/ICRA_2024/ICRA_NEW/output/train_cv_30_entropy_1.0_False',
    #      '/home/jing/Downloads/my_data/cv', nth=1)

    main('cv', '/home/jing/Downloads/ICRA_2024/ICRA_NEW/output/train_cv_30_entropy_10.0_False',
         '/home/jing/Downloads/my_data/cv', nth='3_semi')
