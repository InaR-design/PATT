import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torchvision.models.resnet import Bottleneck, ResNet

import math

class ResNet_ImageNet(ResNet):
    def __init__(self, block, num_blocks, num_classes=1000, return_features=False):
        super(ResNet_ImageNet, self).__init__(block, num_blocks, num_classes=num_classes)
        self.return_features = return_features
        self.penultimate_layer_dim = self.fc.weight.shape[1]
        print('self.penultimate_layer_dim:', self.penultimate_layer_dim)
        
        feat_dim = 1024
        tau = 16.0
        num_head = 2
        self.scale = tau / num_head   # 16.0 / num_head
        self.num_head = num_head
        self.head_dim = feat_dim // num_head

        self.finalScore = nn.Parameter(torch.zeros(feat_dim), requires_grad=False)
        self.head = nn.Sequential(nn.Linear(2048, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(inplace=True),
                    nn.Linear(2048, feat_dim))
    
    def forward_Score(self):
        scaler = MinMaxScaler(feature_range=(0,2))
        finalScore = scaler.fit_transform(self.finalScore.cpu().data.numpy().reshape(-1, 1)).ravel()
        self.finalScore.data = torch.from_numpy(finalScore).cuda()
        return self.finalScore.data, self.linear.bias
    
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def Classbalanced_Calibration(self, x, ood_x, ood_labels, labels, priors, batchs, num_classes, add_inputs=None):
        normed_x = x
        normed_w = self.linear.weight
        
        Score = torch.zeros(512).cuda()
        ood_Score = torch.zeros(512).cuda()
        p = priors**-1
        # p = weight
        #p = torch.cat((priors**-1, torch.tensor([1]).cuda()), dim = 0)
        for feature, label in zip(normed_x, labels):
            Score = torch.mul(feature, normed_w[label]) * p[label] + Score
        Score = Score / len(labels)#一个batch的均值
        self.finalScore.data = self.finalScore.data*batchs + Score

        for feature, label in zip(ood_x, ood_labels):
            ood_Score = torch.mul(feature, normed_w[label]) * p[int(num_classes/2)] + ood_Score#int(num_classes/2)
        ood_Score = ood_Score / len(ood_labels)#一个batch的均值
        self.finalScore.data = self.finalScore.data - ood_Score
    
    def oe_loss_fn(self, ood_logits):
        # pro_logits = self.forward_classifier(self.OOD_pro)
        # logits = torch.cat((logits, pro_logits), dim=0)
        return -(ood_logits.mean(1) - torch.logsumexp(ood_logits, dim=1)).mean()
    
    def multi_head_call(self, func, x):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / (torch.norm(x, 2, 1, keepdim=True) + 1e-12)
        return normed_x
    
    def forward_features(self, x):
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        h1 = self.layer1(c1) # (64,32,32)
        h2 = self.layer2(h1) # (128,16,16)
        h3 = self.layer3(h2) # (256,8,8)
        h4 = self.layer4(h3) # (512,4,4)
        p4 = self.avgpool(h4) # (512,1,1)
        p4 = torch.flatten(p4, 1) # 2048
        return p4

    def forward_classifier(self, p4):
        logits = self.fc(p4) # (10)
        return logits

    def forward(self, x):
        p4 = self.forward_features(x)
        logits = self.forward_classifier(p4)

        p4_head = self.head(p4)

        if self.return_features:
            return logits, p4_head
        else:
            return logits, p4

def ResNet50(num_classes=1000, return_features=False):
    return ResNet_ImageNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, return_features=return_features)

if __name__ == '__main__':
    from thop import profile
    net = ResNet50(num_classes=10)
    x = torch.randn(1,3,224,224)
    flops, params = profile(net, inputs=(x, ))
    y = net(x)
    print(y.size())
    print('GFLOPS: %.4f, #params: %.4fM' % (flops/1e9, params/1e6)) # GFLOPS: 4.1095, #params: 23.5285M

    bn_parameter_number, fc_parameter_number, all_parameter_number = 0, 0, 0
    for name, p in net.named_parameters():
        if 'bn' in name:
            bn_parameter_number += p.numel()
        if 'fc' in name:
            fc_parameter_number += p.numel()
        if 'projection' not in name:
            all_parameter_number += p.numel()

    all_size = all_parameter_number * 4 /1e6 
    bn_size = bn_parameter_number * 4 /1e6 
    fc_size = fc_parameter_number * 4 /1e6 

    print('all_size: %s MB' % (all_size), 2*all_size)
    print('bn_size: %s MB' % (all_size+bn_size), bn_size)
    print('fc_size: %s MB' % (all_size+fc_size), fc_size)
    print('both_size: %s MB' % (all_size+bn_size+fc_size))

'''
module.conv1.weight         
module.bn1.weight           
module.bn1.bias             
module.layer1.0.conv1.weight
module.layer1.0.bn1.weight  
module.layer1.0.bn1.bias    
module.layer1.0.conv2.weight
module.layer1.0.bn2.weight  
module.layer1.0.bn2.bias    
module.layer1.0.conv3.weight
module.layer1.0.bn3.weight  
module.layer1.0.bn3.bias                                                                                                                                  
module.layer1.0.downsample.0.weight
module.layer1.0.downsample.1.weight
module.layer1.0.downsample.1.bias
module.layer1.1.conv1.weight
module.layer1.1.bn1.weight  
module.layer1.1.bn1.bias    
module.layer1.1.conv2.weight
module.layer1.1.bn2.weight  
module.layer1.1.bn2.bias           
module.layer1.1.conv3.weight       
module.layer1.1.bn3.weight  
module.layer1.1.bn3.bias    
module.layer1.2.conv1.weight
module.layer1.2.bn1.weight
module.layer1.2.bn1.bias    
module.layer1.2.conv2.weight
module.layer1.2.bn2.weight
module.layer1.2.bn2.bias
module.layer1.2.conv3.weight
module.layer1.2.bn3.weight
module.layer1.2.bn3.bias
module.layer2.0.conv1.weight       
module.layer2.0.bn1.weight       
module.layer2.0.bn1.bias    
module.layer2.0.conv2.weight
module.layer2.0.bn2.weight  
module.layer2.0.bn2.bias    
module.layer2.0.conv3.weight
module.layer2.0.bn3.weight         
module.layer2.0.bn3.bias           
module.layer2.0.downsample.0.weight
module.layer2.0.downsample.1.weight
module.layer2.0.downsample.1.bias
module.layer2.1.conv1.weight
module.layer2.1.bn1.weight  
module.layer2.1.bn1.bias    
module.layer2.1.conv2.weight
module.layer2.1.bn2.weight  
module.layer2.1.bn2.bias    
module.layer2.1.conv3.weight
module.layer2.1.bn3.weight  
module.layer2.1.bn3.bias    
module.layer2.2.conv1.weight
module.layer2.2.bn1.weight  
module.layer2.2.bn1.bias    
module.layer2.2.conv2.weight
module.layer2.2.bn2.weight  
module.layer2.2.bn2.bias    
module.layer2.2.conv3.weight                                                                                                                              
module.layer2.2.bn3.weight
module.layer2.2.bn3.bias    
module.layer2.3.conv1.weight
module.layer2.3.bn1.weight
module.layer2.3.bn1.bias    
module.layer2.3.conv2.weight
module.layer2.3.bn2.weight
module.layer2.3.bn2.bias    
module.layer2.3.conv3.weight       
module.layer2.3.bn3.weight         
module.layer2.3.bn3.bias
module.layer3.0.conv1.weight
module.layer3.0.bn1.weight
module.layer3.0.bn1.bias    
module.layer3.0.conv2.weight
module.layer3.0.bn2.weight
module.layer3.0.bn2.bias    
module.layer3.0.conv3.weight
module.layer3.0.bn3.weight
module.layer3.0.bn3.bias    
module.layer3.0.downsample.0.weight
module.layer3.0.downsample.1.weight
module.layer3.0.downsample.1.bias
module.layer3.1.conv1.weight
module.layer3.1.bn1.weight
module.layer3.1.bn1.bias    
module.layer3.1.conv2.weight
module.layer3.1.bn2.weight
module.layer3.1.bn2.bias           
module.layer3.1.conv3.weight       
module.layer3.1.bn3.weight       
module.layer3.1.bn3.bias    
module.layer3.2.conv1.weight
module.layer3.2.bn1.weight
module.layer3.2.bn1.bias    
module.layer3.2.conv2.weight
module.layer3.2.bn2.weight
module.layer3.2.bn2.bias    
module.layer3.2.conv3.weight
module.layer3.2.bn3.weight
module.layer3.2.bn3.bias    
module.layer3.3.conv1.weight
module.layer3.3.bn1.weight
module.layer3.3.bn1.bias    
module.layer3.3.conv2.weight
module.layer3.3.bn2.weight
module.layer3.3.bn2.bias    
module.layer3.3.conv3.weight
module.layer3.3.bn3.weight
module.layer3.3.bn3.bias
module.layer3.4.conv1.weight
module.layer3.4.bn1.weight
module.layer3.4.bn1.bias
module.layer3.4.conv2.weight
module.layer3.4.bn2.weight
module.layer3.4.bn2.bias  
module.layer3.4.conv3.weight
module.layer3.4.bn3.weight         
module.layer3.4.bn3.bias 
module.layer3.5.conv1.weight
module.layer3.5.bn1.weight
module.layer3.5.bn1.bias
module.layer3.5.conv2.weight
module.layer3.5.bn2.weight
module.layer3.5.bn2.bias
module.layer3.5.conv3.weight
module.layer3.5.bn3.weight
module.layer3.5.bn3.bias
module.layer4.0.conv1.weight
module.layer4.0.bn1.weight
module.layer4.0.bn1.bias
module.layer4.0.conv2.weight
module.layer4.0.bn2.weight
module.layer4.0.bn2.bias
module.layer4.0.conv3.weight
module.layer4.0.bn3.weight
module.layer4.0.bn3.bias
module.layer4.0.downsample.0.weight
module.layer4.0.downsample.1.weight
module.layer4.0.downsample.1.bias
module.layer4.1.conv1.weight
module.layer4.1.bn1.weight
module.layer4.1.bn1.bias
module.layer4.1.conv2.weight
module.layer4.1.bn2.weight
module.layer4.1.bn2.bias
module.layer4.1.conv3.weight
module.layer4.1.bn3.weight
module.layer4.1.bn3.bias
module.layer4.2.conv1.weight
module.layer4.2.bn1.weight
module.layer4.2.bn1.bias
module.layer4.2.conv2.weight
module.layer4.2.bn2.weight
module.layer4.2.bn2.bias
module.layer4.2.conv3.weight
module.layer4.2.bn3.weight
module.layer4.2.bn3.bias
module.fc.weight
module.fc.bias
module.projection.0.weight
module.projection.0.bias
module.projection.2.weight
module.projection.2.bias
'''