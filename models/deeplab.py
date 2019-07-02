import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
import numpy as np

from models.backbone import build_backbone

def _make_pred_layer(block, dilation_series, padding_series,NoLabels):
    return block(dilation_series,padding_series,NoLabels)

class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)


    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class MS_Deeplab(nn.Module):
    def __init__(self, NoLabels, pretrained=False):
        super(MS_Deeplab,self).__init__()
        self.backbone= build_backbone('resnet', in_channel=4, pretrained=pretrained)
        self.classifier = _make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24], NoLabels)

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406, 0]).view(1,4,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225, 0.358]).view(1,4,1,1))

    
    def forward(self, x):
        x = (x - self.mean) / self.std
        out = self.backbone(x)
        out = self.classifier(out)

        return out

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.backbone.conv1)
        b.append(self.backbone.bn1)
        b.append(self.backbone.layer1)
        b.append(self.backbone.layer2)
        b.append(self.backbone.layer3)
        b.append(self.backbone.layer4)

        
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """

        b = []
        b.append(self.classifier.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

class Deeplab_fuse(nn.Module):
    def __init__(self, NoLabels, pretrained):
        super(Deeplab_fuse,self).__init__()
        self.backbone= build_backbone('resnet_ms', in_channel=3, pretrained=pretrained)
        self.aspp = build_aspp(output_stride=16)

        self.branch = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        self.fuse = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.ReLU())

        self.refine= nn.Sequential(
                #nn.Conv2d(48+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(256+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU())

        self.predict = nn.Sequential(
                #nn.Conv2d(48+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(128+64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, NoLabels, kernel_size=1))
        

        #self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406, -0.329]).view(1,4,1,1))
        #self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225, 0.051]).view(1,4,1,1))
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, mask):
        x = (x - self.mean) / self.std
        ll, low_level_feat, out,  = self.Scale(x)
        mask = F.interpolate(mask, size=out.shape[2:])

        branch_feature = self.branch(out)
        mask_feature = branch_feature * mask
        fused_feature = torch.cat([branch_feature, mask_feature], 1)
        fused_feature = self.fuse(fused_feature)
        out = out + fused_feature

        out = self.aspp(out)
        #branch_feature = self.branch(low_level_feat)

        
        #mask = F.interpolate(mask, size=branch_feature.shape[2:])
        #mask_feature = branch_feature * mask
        #fused_feature = torch.cat([branch_feature, mask_feature], 1)
        #fused_feature = self.fuse(fused_feature)
        #branch_feature = branch_feature + fused_feature

        out = F.interpolate(out, size=low_level_feat.shape[2:])
        out = torch.cat([out, low_level_feat], 1)
        out = self.refine(out)
        out = F.interpolate(out, size=ll.shape[2:])
        out = torch.cat([out, ll], 1)
        out = self.predict(out)

        return out

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.backbone.conv1)
        b.append(self.backbone.bn1)
        b.append(self.backbone.layer1)
        b.append(self.backbone.layer2)
        b.append(self.backbone.layer3)
        b.append(self.backbone.layer4)
        b.append(self.branch)
        b.append(self.fuse)
        b.append(self.refine)

        
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """

        b = []
        b.append(self.predict.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

def build_Deeplab(NoLabels=2):
    model = MS_Deeplab(NoLabels, pretrained=True)
    return model

