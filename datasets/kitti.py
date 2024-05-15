from datasets.dataset_template import DatasetTemplate,get_label_image_path
from tool.build_dataset.kitti import index_label
from torch.utils.data import DataLoader
import numpy as np


class KITTI(DatasetTemplate):
    def __init__(self, mode, dataset_path, used_class, fully_supervised=False, ignore_index=255):
        super().__init__()

        self.mean, self.std, self.crop_size = [0.3790, 0.3984, 0.3836], [0.3101, 0.3190, 0.3282], [375, 1242]
        self.mean_val = tuple((np.array(self.mean) * 255.0).astype(np.uint8).tolist())
        self.mode = mode
        self.fully_supervised = fully_supervised
        self.ignore_index = ignore_index

        self.list_inputs, self.list_labels = get_label_image_path(dataset_path, mode, index_label, used_class)


if __name__ == "__main__":
    dataset_ = KITTI("val", "/home/jing/Downloads/my_data/kitti_semantic", 1, fully_supervised=True)
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