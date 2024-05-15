import torch
import torch.nn as nn


class SegmentHead(nn.Module):

    def __init__(self, class_num):
        super(SegmentHead, self).__init__()
        self.segment_head = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(), )

        self.classifier = nn.Conv2d(256, class_num, 1)
        self._initialize_weights()
        self.class_num = class_num

    def forward(self, x):
        emb = self.segment_head(x)
        pred = self.classifier(emb)
        return {"emb": emb, "pred": pred}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    x = torch.rand(1, 304, 14, 14)
    model = SegmentHead(19)
    output = model(x)
    print(output["pred"].size())

