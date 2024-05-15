from tool.utils import *
from tqdm import tqdm
import torch.nn.functional as F
from tool.query import *

if __name__ == "__main__":
    dataset_path = "/home/jing/Downloads/my_data/cs"
    dataset = 'cs'
    index_label = used_classes(dataset)
    result = {x: 0 for x in index_label.keys()}
    datasets_train = define_dataset(dataset, 'train', dataset_path, None, fully_supervised=True)
    # datasets_val = define_dataset(dataset, 'val', dataset_path, None)
    #
    # dataloader_val = get_dataloader(datasets_val, 6, 4, shuffle_=False)
    dataloader_train = get_dataloader(datasets_train, 8, 4, shuffle_=True)

    dataloader_iter, tbar = iter(dataloader_train), tqdm(range(len(dataloader_train)))
    for _ in tbar:
        dict_data = next(dataloader_iter)
        y = dict_data['y']

        for k, v in result.items():
            num = torch.sum(y == k)
            result[k] += num

    final_result = {k: (v / (len(dataloader_train) * 8 * 360 * 480)).numpy() * 100 for k, v in result.items()}

    print(final_result)
