import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.aspp import ASPP
from networks.decoder import SegmentHead
from networks.mobilenet_v2 import MobileNetV2


class Deeplab(nn.Module):

    def __init__(self, class_num, output_stride=16, pretrained=True):
        super(Deeplab, self).__init__()
        self.backbone = MobileNetV2(output_stride, nn.BatchNorm2d, pretrained)
        self.aspp = ASPP(output_stride, nn.BatchNorm2d)

        low_level_channel = 24  # [0:4]
        self.low_level_conv = nn.Sequential(nn.Conv2d(low_level_channel, 48, 1, bias=False),
                                            nn.BatchNorm2d(48),
                                            nn.ReLU())
        self.seg_head = SegmentHead(class_num)

    def forward(self, inputs):
        backbone_feat, low_level_feat = self.backbone(inputs)
        x = self.aspp(backbone_feat)

        low_level_feat = self.low_level_conv(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode="bilinear", align_corners=True)
        second_to_last_features = torch.cat((x, low_level_feat), dim=1)

        dict_outputs = self.seg_head(second_to_last_features)

        pred = F.interpolate(dict_outputs["pred"], size=inputs.size()[2:], mode="bilinear", align_corners=True)
        dict_outputs["pred"] = pred

        emb = F.interpolate(dict_outputs["emb"], size=inputs.size()[2:], mode="bilinear", align_corners=True)
        dict_outputs["emb"] = emb
        dict_outputs["backbone_feat"] = second_to_last_features

        return dict_outputs


if __name__ == '__main__':
    # x = torch.rand(2, 3, 512, 512)
    # model = Deeplab(19, 16)
    # output = model(x)
    # print(output['pred'].shape)
    # print(output['emb'].shape)
    from tool.utils import get_dataloader
    from datasets.eval_dataset import Dataset_eval
    import cv2
    import numpy as np

    dataset_ = Dataset_eval(dataset_name='cs', mode='val',
                            dir_dataset='/mnt/Jan/Download/Dataset/Extracted_Data/cs/cityscapes')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(dataset_, 1, 4, False)
    model = Deeplab(2).to(device)
    pt_weight = "/mnt/Jan/Download/master_thesis/testing/Uncertainty_ratio/UC_0_13_11_EP_20_MB_100_IR_random_P_0.1_NRN_9_SS_/Uncertainty_Ratio_0.1/road/9_query/final_epoch_model.pt"
    state_dict: dict = torch.load(pt_weight, map_location=device)["model"]
    model.load_state_dict(state_dict)
    model.eval()

    # dataloader_iter = iter(dataloader)
    with torch.no_grad():
        for index_num, _ in enumerate(dataloader):
            if index_num < 38:
                pass
            else:
                dict_data = _
                x = dict_data['x'].to(device)
                print(dict_data['p_img'])
                dict_outputs = model(x)


                def scale_to_01_range(x):
                    value_range = (torch.max(x) - torch.min(x))

                    starts_from_zero = x - torch.min(x)

                    return starts_from_zero / value_range


                backbone_out = scale_to_01_range(dict_outputs["backbone_feat"])
                backbone_out = backbone_out.permute(0, 2, 3, 1)
                image = backbone_out[0].cpu().numpy()
                image_show = image[:, :, 0]
                for i in range(image.shape[2])[1:257]:
                    image_show += image[:, :, i]
                    # image_show[image_show>0.1] = 1
                    # image_show[image_show <= 0.1] = 0

                image_show = (image_show - np.min(image_show)) / (np.max(image_show) - np.min(image_show))
                image_show = np.clip(image_show * 255, 0, 255).astype(np.uint8)

                cv2.imwrite(f"/mnt/Jan/Download/master_thesis/fig_web/features/road/road_2.png", image_show)

                break
