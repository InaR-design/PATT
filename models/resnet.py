''' PyTorch implementation of ResNet taken from 
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
which is originally licensed under MIT.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.preprocessing import MinMaxScaler


class BasicBlock(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, norm, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                norm(out_planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # print(out.size())
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, pooling='avgpool', norm=nn.BatchNorm2d, return_features=False):
        super(ResNet, self).__init__()
        if pooling == 'avgpool':
            self.pooling = nn.AvgPool2d(4)
        elif pooling == 'maxpool':
            self.pooling = nn.MaxPool2d(4)
        else:
            raise Exception('Unsupported pooling: %s' % pooling)
        self.in_planes = 64
        self.return_features = return_features

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm=norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm=norm)
        self.linear = nn.Linear(512, num_classes)
        
        self.finalScore = nn.Parameter(torch.zeros(512), requires_grad=False)

        feat_dim = 256
        tau = 16.0
        num_head = 2
        self.scale = tau / num_head   # 16.0 / num_head
        self.num_head = num_head
        self.head_dim = feat_dim // num_head

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
    
    def return_weighs(self):
        return self.weight.detach().cpu().numpy().squeeze()

    def _make_layer(self, block, planes, num_blocks, norm, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, planes, norm, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x):
        c1 = F.relu(self.bn1(self.conv1(x))) # (3,32,32)
        h1 = self.layer1(c1) # (64,32,32)
        h2 = self.layer2(h1) # (128,16,16)
        h3 = self.layer3(h2) # (256,8,8)
        h4 = self.layer4(h3) # (512,4,4)
        p4 = self.pooling(h4) # (512,1,1)
        p4 = p4.view(p4.size(0), -1) # (512)
        return p4
    
    def forward_Score(self):
        scaler = MinMaxScaler(feature_range=(0,2))
        finalScore = scaler.fit_transform(self.finalScore.cpu().data.numpy().reshape(-1, 1)).ravel()
        self.finalScore.data = torch.from_numpy(finalScore).cuda()
        return self.finalScore.data


    def forward_classifier(self, p4):
        logits = self.linear(p4) # (10)
        return logits

    def forward(self, x):
        p4 = self.forward_features(x)
        logits = self.forward_classifier(p4)

        if self.return_features:
            return logits, p4
        else:
            return logits
    
    # Post-hoc feature calibration
    def Classbalanced_Calibration(self, x, ood_x, ood_labels, labels, priors, batchs, num_classes, add_inputs=None):
        normed_x = x
        normed_w = self.linear.weight
        
        Score = torch.zeros(512).cuda()
        ood_Score = torch.zeros(512).cuda()
        p = priors**-1
        for feature, label in zip(normed_x, labels):
            Score = torch.mul(feature, normed_w[label]) * p[label] + Score

        Score = Score / len(labels)
        self.finalScore.data = self.finalScore.data*batchs + Score

        for feature, label in zip(ood_x, ood_labels):
            ood_Score = torch.mul(feature, normed_w[label]) * p[label] + ood_Score
        ood_Score = ood_Score / len(ood_labels)
        self.finalScore.data = self.finalScore.data - ood_Score


    
    def oe_loss_fn(self, ood_logits):
        return -(ood_logits.mean(1) - torch.logsumexp(ood_logits, dim=1)).mean()

    def multi_head_call(self, func, x):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 2, keepdim=True)
        return normed_x


def ResNet18(num_classes=10, pooling='avgpool', norm=nn.BatchNorm2d, return_features=False):
    '''
    GFLOPS: 0.5579, model size: 11.1740MB
    '''
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, pooling=pooling, norm=norm, return_features=return_features)

def ResNet34(num_classes=10, pooling='avgpool', norm=nn.BatchNorm2d, return_features=False):
    '''
    GFLOPS: 1.1635, model size: 21.2859MB
    '''
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, pooling=pooling, norm=norm, return_features=return_features)



if __name__ == '__main__':
    from thop import profile
    net = ResNet18(num_classes=10, return_features=True)
    x = torch.randn(1,3,32,32)
    flops, params = profile(net, inputs=(x, ))
    y, features = net(x)
    print(y.size())
    print('GFLOPS: %.4f, model size: %.4fMB' % (flops/1e9, params/1e6))

'''
conv1.weight       
bn1.weight       
bn1.bias                  
layer1.0.conv1.weight                                                                                                                                     
layer1.0.bn1.weight     
layer1.0.bn1.bias    
layer1.0.conv2.weight
layer1.0.bn2.weight
layer1.0.bn2.bias    
layer1.1.conv1.weight
layer1.1.bn1.weight
layer1.1.bn1.bias
layer1.1.conv2.weight
layer1.1.bn2.weight
layer1.1.bn2.bias
layer2.0.conv1.weight
layer2.0.bn1.weight
layer2.0.bn1.bias  
layer2.0.conv2.weight
layer2.0.bn2.weight                
layer2.0.bn2.bias
layer2.0.shortcut.0.weight
layer2.0.shortcut.1.weight
layer2.0.shortcut.1.bias
layer2.1.conv1.weight                 
layer2.1.bn1.weight                                                          
layer2.1.bn1.bias                                                            
layer2.1.conv2.weight                                                        
layer2.1.bn2.weight                                                          
layer2.1.bn2.bias                                                                                                                                         
layer3.0.conv1.weight                 
layer3.0.bn1.weight                                                          
layer3.0.bn1.bias                                                            
layer3.0.conv2.weight                                                        
layer3.0.bn2.weight                                                          
layer3.0.bn2.bias                                                                                                                                         
layer3.0.shortcut.0.weight            
layer3.0.shortcut.1.weight         
layer3.0.shortcut.1.bias                                                     
layer3.1.conv1.weight                                                                                                                                     
layer3.1.bn1.weight
layer3.1.bn1.bias 
layer3.1.conv2.weight                
layer3.1.bn2.weight             
layer3.1.bn2.bias                    
layer4.0.conv1.weight          
layer4.0.bn1.weight                                                          
layer4.0.bn1.bias                                                                                                                                         
layer4.0.conv2.weight                                                        
layer4.0.bn2.weight
layer4.0.bn2.bias
layer4.0.shortcut.0.weight
layer4.0.shortcut.1.weight
layer4.0.shortcut.1.bias
layer4.1.conv1.weight
layer4.1.bn1.weight  
layer4.1.bn1.bias  
layer4.1.conv2.weight
layer4.1.bn2.weight  
layer4.1.bn2.bias  
linear.weight    
linear.bias          
projection.0.weight  
projection.0.bias  
projection.2.weight
projection.2.bias 
'''