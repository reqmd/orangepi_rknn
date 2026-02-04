import torch
import torch.nn as nn
import numpy as np

class ShuffleNetStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        branch_features = out_channels // 2

        if self.stride == 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, 3, stride, 1, groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features)
            )
            self.branch2 = nn.Sequential()
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, 3, stride, 1, groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, 3, stride, 1, groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features)
            )
    
    def _channel_shuffle(self, X, groups):
        batch_size, num_channels, H, W = X.data.size()
        channels_pre_group = num_channels // groups
        X = X.view(batch_size, groups, channels_pre_group, H, W)
        X = torch.transpose(X, 1, 2).contiguous()
        X = X.view(batch_size, -1, H, W)
        return X
    
    def forward(self, X):
        if self.stride == 1:
            X1, X2 = X.chunk(2, dim = 1)
            out = torch.cat((X1, self.branch1(X2)), dim=1)
        else:
            out = torch.cat((self.branch1(X), self.branch2(X)), dim = 1)
        return self._channel_shuffle(out, 2)
    

class ShuffleNet(nn.Module):
    def __init__(self, alpha = 1, stages_repeats = [4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024], num_classes=1000):
        super().__init__()
        self.stages_repeats = stages_repeats
        self.stages_out_channels = stages_out_channels

        self.stages_out_channels = [round(i * alpha) for i in self.stages_out_channels]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.stages_out_channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.stages_out_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.stage2 = self._make_stage(in_channels=self.stages_out_channels[0], 
                                       out_channels=self.stages_out_channels[1], 
                                       repeats=self.stages_repeats[0], stride=2)
        
        self.stage3 = self._make_stage(in_channels=self.stages_out_channels[1], 
                                       out_channels=self.stages_out_channels[2], 
                                       repeats=self.stages_repeats[1], stride=2)
        
        self.stage4 = self._make_stage(in_channels=self.stages_out_channels[2], 
                                       out_channels=self.stages_out_channels[3], 
                                       repeats=self.stages_repeats[2], stride=2)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.stages_out_channels[3], self.stages_out_channels[4], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stages_out_channels[4]),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(self.stages_out_channels[4], num_classes)


    def _make_stage(self, in_channels, out_channels, repeats, stride):
        layers = []
        layers.append(ShuffleNetStage(in_channels=in_channels, out_channels=out_channels, stride=stride))
        for _ in range(repeats - 1):
            layers.append(ShuffleNetStage(in_channels=out_channels, out_channels=out_channels, stride=stride))
        return nn.Sequential(*layers)
    
    def forward(self, X):
        x = self.conv1(X)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x
    

