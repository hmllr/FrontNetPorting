
import torch.nn as nn
from PreActBlockSimple import PreActBlockSimple
from nemo.quant.pact_quant import PACT_Conv1d, PACT_Conv2d, PACT_Linear, PACT_Act, PACT_ThresholdAct




def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



# FrontNet
class FindNet(nn.Module):
    def __init__(self, block, layers, isGray=False, isClassifier=False, isCombined=False):
        super(FindNet, self).__init__()

        self.isClassifier = isClassifier
        self.isCombined = isCombined
        if isGray ==True:
            self.name = "FindNetGray_classifier_2res_alldario_himaxaugmented"
        else:
            self.name = "FindNetRGB"
        self.inplanes = 32
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d

        replace_stride_with_dilation = [False, False, False]

        self.groups = 1
        self.base_width = 64
        if isGray == True:
            self.conv = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        else:
            self.conv = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)

        self.bn32_1 = nn.BatchNorm2d(32)
        self.bn32_2 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.layer1 = PreActBlockSimple(32, 32, stride=1)
        self.layer2 = PreActBlockSimple(32, 64, stride=2)
        self.layer3 = PreActBlockSimple(64, 128, stride=2)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 7), stride=(1, 1))
        #self.fc1 = nn.Linear(128 * block.expansion, 128)
        #self.fc2 = nn.Linear(128, 64)

        self.fc_x = nn.Linear(64, 1)
        self.fc_y = nn.Linear(64, 1)
        self.fc_z = nn.Linear(64, 1)
        self.fc_phi = nn.Linear(64, 1)

        self.fc_class = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

        self.dropout = nn.Dropout()


    def forward(self, x):
        out = self.conv(x)
        out = self.bn32_1(out)
        out = self.relu1(out)
        out = self.maxpool(out)
        #out = self.layer1(out)
        #out = self.bn32_2(out)
        #out = self.relu2(out)
        #out = self.dropout(out)
        out = self.layer2(out)
        out = self.bn64(out)
        out = self.relu3(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        #out = self.fc1(out)
        #out = self.relu4(out)
        #out = self.fc2(out)
        #out = self.relu5(out)
        #x = self.fc_x(out)
        #y = self.fc_y(out)
        #z = self.fc_z(out)
        #phi = self.fc_phi(out)
        out = self.dropout(out)
        head = self.fc_class(out)
        if self.isCombined:
            return [x, y, z, phi, self.sig(head)]
        elif self.isClassifier:
            return self.sig(head)

        return [x, y, z, phi]

