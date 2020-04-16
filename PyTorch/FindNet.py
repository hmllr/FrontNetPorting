
import torch.nn as nn
from PreActBlockSimple import PreActBlockSimple
from nemo.quant.pact_quant import PACT_Conv1d, PACT_Conv2d, PACT_Linear, PACT_Act, PACT_ThresholdAct
import numpy as np




def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



# FrontNet
class FindNet(nn.Module):
    def __init__(self, block, layers, isGray=False, isClassifier=False, isCombined=False):
        super(FindNet, self).__init__()
        self.i = 0

        self.isClassifier = isClassifier
        self.isCombined = isCombined
        if isGray ==True:
            self.name = "FindNetGray_classifier_3_onlyhimax_extraBNRELU"
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
        self.bn128 = nn.BatchNorm2d(128)

        self.relu1 = nn.ReLU(inplace=True)

        
        #self.relu4 = nn.ReLU(inplace=True)
        #self.relu5 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.layer1 = PreActBlockSimple(32, 32, stride=1, shortcut=False)
        self.relu2 = nn.ReLU(inplace=True)
        #self.dropout1 = nn.Dropout()
        self.layer2 = PreActBlockSimple(32, 64, stride=2, shortcut=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        #self.dropout2 = nn.Dropout()
        self.layer3 = PreActBlockSimple(64, 128, stride=2, shortcut=False)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 7), stride=(1, 1))
        #self.fc1 = nn.Linear(128 * block.expansion, 128)
        #self.fc2 = nn.Linear(128, 64)

        #self.fc_x = nn.Linear(64, 1)
        #self.fc_y = nn.Linear(64, 1)
        #self.fc_z = nn.Linear(64, 1)
        #self.fc_phi = nn.Linear(64, 1)
        #self.dropout3 = nn.Dropout()
        self.fc_class = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()
        #self.print= False



    def forward(self, x):
        self.i += 1
        # print("x:",x[0][0][0])
        out = self.conv(x)
        # print("c1:",out[0][0][0])
        out = self.bn32_1(out)
        # print("bn1:",out[0][0][0])
        out = self.relu1(out)
        # print("relu1:",out[0][0][0])
        #if self.print:
        #    np.savetxt("frontnet/after_relu1%d.txt" % self.i, out.cpu().detach().numpy().flatten(), header="layer 3 output (batch %d)" % (self.i), fmt="%.3f", delimiter=',', newline=',\n')
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.bn32_2(out)
        out = self.relu2(out)
        #print("l1:",out[0][0][0])
        #np.savetxt("frontnet/after_relu2%d.txt" % self.i, out.cpu().detach().numpy().flatten(), header="layer 3 output (batch %d)" % (self.i), fmt="%.3f", delimiter=',', newline=',\n')
        #out = self.dropout1(out)
        out = self.layer2(out)
        out = self.bn64(out)
        out = self.relu3(out)
        #print("l2:",out[0][0][0])
        #np.savetxt("frontnet/after_relu3%d.txt" % self.i, out.cpu().detach().numpy().flatten(), header="layer 3 output (batch %d)" % (self.i), fmt="%.3f", delimiter=',', newline=',\n')
        #out = self.dropout2(out)
        out = self.layer3(out)
        out = self.bn128(out)
        out = self.relu4(out)
        #print("l3:",out[0][0][0])
        #np.savetxt("frontnet/before_avg%d.txt" % self.i, out.cpu().detach().numpy().flatten(), header="layer 3 output (batch %d)" % (self.i), fmt="%.3f", delimiter=',', newline=',\n')
        out = self.avg_pool(out)
        #print("avg:",out[0][0])

        # out = out.view(out.size(0), -1)
        out = out.flatten(1)
        #np.savetxt("frontnet/after_avg%d.txt" % self.i, out.cpu().detach().numpy().flatten(), header="layer avg output (batch %d)" % (self.i), fmt="%.3f", delimiter=',', newline=',\n')
        #out = self.dropout3(out)
        head = self.fc_class(out)
        #print("head:", head)
        if self.isCombined:
            return [x, y, z, phi, self.sig(head)]
        elif self.isClassifier:
            out = self.sig(head)
            #print(out[0])
            #np.savetxt("frontnet/sigmoid%d.txt" % self.i, head.cpu().detach().numpy().flatten(), header="sigmoid output (batch %d)" % (self.i), fmt="%.3f", delimiter=',', newline=',\n')
            return out

        return [x, y, z, phi]


