from tool.utils import *
from tqdm import tqdm
import torch.nn.functional as F
from tool.query import *
from networks.deeplab import Deeplab


def train_(model, dataloader_train, device, model_support, optimizer, lr_scheduler, epoch):
    model.train()
    dataloader_iter, tbar = iter(dataloader_train), tqdm(range(len(dataloader_train)))
    uncertainty_sampler = UncertaintySampler('entropy')
    for _ in tbar:
        dict_data = next(dataloader_iter)
        x = dict_data["x"].to(device)

        dict_outputs = model(x)
        dict_outputs_support = model_support(x)
        logits_support = dict_outputs_support["pred"]
        logits = dict_outputs["pred"]

        prob_support, pred_support = F.softmax(logits_support.detach(), dim=1), logits_support.argmax(dim=1)
        unc_map = uncertainty_sampler(prob_support)
        unc_map_flatten = unc_map.flatten()

        topk_value = torch.max(unc_map_flatten.topk(k=1000, dim=0, largest=True).values)
        pred_support[unc_map > topk_value] = ~pred_support[unc_map > topk_value]

        loss_cross_entropy = F.cross_entropy(logits, pred_support, ignore_index=255)

        optimizer.zero_grad()
        loss_cross_entropy.backward()
        optimizer.step()

    lr_scheduler.step(epoch=epoch - 1)
    return model, optimizer, lr_scheduler


def _val(model, epoch, dataloader_val, device, epochs, training_class_nth_dir):
    model.eval()
    dataloader_iter, tbar = iter(dataloader_val), tqdm(range(len(dataloader_val)))
    running_score = RunningScore(2)
    with torch.no_grad():
        for _ in tbar:
            dict_data = next(dataloader_iter)
            # y b x 1 x w x h
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            dict_outputs = model(x)
            logits = dict_outputs['pred']
            pred = logits.argmax(dim=1)

            running_score.update(y.cpu().numpy(), pred.cpu().numpy())
    if epoch == epochs - 1:
        state_dict = {"model": model.state_dict()}
        torch.save(state_dict, f"{training_class_nth_dir}/final_epoch_model.pt")
    scores = running_score.get_scores()
    print(f"Training_class: {label_name} Epoch : {epoch} | Foreground IoU : {scores['IoU'][1]:.4f} "
          f"Background IoU : {scores['IoU'][0]:.4f} | ")


if __name__ == "__main__":
    training_folder = "/home/jing/Downloads/ICRA_2024/ICRA_NEW/output/train_cv_30_entropy_10.0_False"
    dataset_path = "/home/jing/Downloads/my_data/cv"
    nth_list = [3]
    dataset = 'cv'
    device = 'cuda:1'
    epochs = 30
    index_label = used_classes(dataset)
    for nth_ in nth_list:
        nth_semi = f"{training_folder}/{nth_}_semi"
        os.makedirs(nth_semi, exist_ok=True)
        for index, label_name in index_label.items():
            nth_semi_category = f"{nth_semi}/{label_name}"
            os.makedirs(nth_semi_category, exist_ok=True)
            the_previous_weights = f"{training_folder}/{nth_}_th/{label_name}/final_epoch_model.pt"
            datasets_train = define_dataset(dataset, 'train', dataset_path, index)
            datasets_val = define_dataset(dataset, 'val', dataset_path, index)

            if nth_ == 0:
                query_dict = read_pkl(training_folder + f"/{nth_}_th/queries.pkl")
            else:
                query_dict = read_pkl(training_folder + f"/{nth_}_th/{label_name}_queries.pkl")


            datasets_train.update_queries(query_dict)

            dataloader_val = get_dataloader(datasets_val, 6, 4, shuffle_=False)
            dataloader_train = get_dataloader(datasets_train, 6, 4, shuffle_=True)

            model = Deeplab(class_num=2).to(device)
            model_support = Deeplab(class_num=2).to(device)
            if the_previous_weights is not None:
                state_dict: dict = torch.load(the_previous_weights, map_location=device)["model"]
                model.load_state_dict(state_dict)
                model_support.load_state_dict(state_dict)
                print(f"load {the_previous_weights}")

            optimizer = get_optimizer(model, case_=dataset)
            lr_scheduler = get_lr_scheduler(epochs, optimizer, len(dataloader_train),
                                            lr_scheduler_type='Poly')

            for e in range(epochs):
                model, optimizer, lr_scheduler = train_(model, dataloader_train, device, model_support, optimizer,
                                                        lr_scheduler, e)
                _val(model, e, dataloader_val, device, epochs, nth_semi_category)
