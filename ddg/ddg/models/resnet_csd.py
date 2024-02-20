from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
from resnet import *
import torch
import numpy as np

class ResNetCSD(ResNet):
    def __init__(self, has_fc, num_classes=100):
        super(ResNetCSD, self).__init__(num_classes=num_classes)
        self.has_fc = has_fc


        K=2
        self.sms = torch.nn.Parameter(torch.normal(0, 1e-1, size=[K, 512, num_classes], dtype=torch.float, device='cuda'), requires_grad=True)
        self.sm_biases = torch.nn.Parameter(torch.normal(0, 1e-1, size=[K, num_classes], dtype=torch.float, device='cuda'), requires_grad=True)
    
        self.embs = torch.nn.Parameter(torch.normal(mean=0., std=1e-4, size=[3, K-1], dtype=torch.float, device='cuda'), requires_grad=True)
        self.cs_wt = torch.nn.Parameter(torch.normal(mean=.1, std=1e-4, size=[], dtype=torch.float, device='cuda'), requires_grad=True)
    
    
    def _forward_impl(self, x, uids, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)


        w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
        # 8th Layer: FC and return unscaled activations
        logits_common = torch.matmul(x, w_c) + b_c

        c_wts = torch.matmul(uids, self.embs)
        # B x K
        batch_size = uids.shape[0]
        c_wts = torch.cat((torch.ones((batch_size, 1), dtype=torch.float, device='cuda')*self.cs_wt, c_wts), 1)
        c_wts = torch.tanh(c_wts)
        w_d, b_d = torch.einsum("bk,krl->brl", c_wts, self.sms), torch.einsum("bk,kl->bl", c_wts, self.sm_biases)
        logits_specialized = torch.einsum("brl,br->bl", w_d, x) + b_d
        
        return logits_specialized, logits_common
        # return super.self.fc(x)


# class ResNetCSD(nn.Module):
#     def __init__(self, block, layers, jigsaw_classes=1000, classes=100, domains=3):
#         self.inplanes = 64
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
    
#         K = 2
#         self.sms = torch.nn.Parameter(torch.normal(0, 1e-1, size=[K, 512, classes], dtype=torch.float, device='cuda'), requires_grad=True)
#         self.sm_biases = torch.nn.Parameter(torch.normal(0, 1e-1, size=[K, classes], dtype=torch.float, device='cuda'), requires_grad=True)
    
#         self.embs = torch.nn.Parameter(torch.normal(mean=0., std=1e-4, size=[3, K-1], dtype=torch.float, device='cuda'), requires_grad=True)
#         self.cs_wt = torch.nn.Parameter(torch.normal(mean=.1, std=1e-4, size=[], dtype=torch.float, device='cuda'), requires_grad=True)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def is_patch_based(self):
#         return False

#     def forward(self, x, uids, **kwargs):
#         # one hot uids
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
        
#         w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
#         # 8th Layer: FC and return unscaled activations
#         logits_common = torch.matmul(x, w_c) + b_c

#         c_wts = torch.matmul(uids, self.embs)
#         # B x K
#         batch_size = uids.shape[0]
#         c_wts = torch.cat((torch.ones((batch_size, 1), dtype=torch.float, device='cuda')*self.cs_wt, c_wts), 1)
#         c_wts = torch.tanh(c_wts)
#         w_d, b_d = torch.einsum("bk,krl->brl", c_wts, self.sms), torch.einsum("bk,kl->bl", c_wts, self.sm_biases)
#         logits_specialized = torch.einsum("brl,br->bl", w_d, x) + b_d
        
#         return logits_specialized, logits_common


# def resnet18(pretrained=True, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNetCSD(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
#     return model

# def resnet50(pretrained=True, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNetCSD(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
#     return model