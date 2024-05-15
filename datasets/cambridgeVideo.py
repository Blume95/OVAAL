from datasets.dataset_template import DatasetTemplate,get_label_image_path
from tool.build_dataset.camvid import index_label
from torch.utils.data import DataLoader
import numpy as np


class CamVid(DatasetTemplate):
    def __init__(self, mode, dataset_path, used_class, fully_supervised=False, ignore_index=255):
        super().__init__()

        self.mean, self.std, self.crop_size = [0.41189489566336, 0.4251328133025, 0.4326707089857], [0.27413549931506,
                                                                                                     0.28506257482912, 0.28284674400252], [360, 480]
        self.mean_val = tuple((np.array(self.mean) * 255.0).astype(np.uint8).tolist())
        self.mode = mode
        self.fully_supervised = fully_supervised
        self.ignore_index = ignore_index

        self.list_inputs, self.list_labels = get_label_image_path(dataset_path, mode, index_label, used_class)


if __name__ == "__main__":
    dataset_ = CamVid("val", "/home/jing/Downloads/my_data/cv", 1, fully_supervised=True)
    dataloader = DataLoader(
        dataset_,
        batch_size=4,
        num_workers=4,
        shuffle=True,
        drop_last=len(dataset_) % 4 == 1
    )
    for data in dataloader:
        print(data['x'].shape)
        print(data['y'].shape)