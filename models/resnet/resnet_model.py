"""
1D ResNet 기반 시계열 주가 예측 모델 아키텍처
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BasicBlock1D(nn.Module):
    """1D ResNet Basic Block for Time Series"""
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetForPricePrediction(nn.Module):
    """시계열 주가 예측을 위한 1D ResNet 모델"""
    
    def __init__(self, block=BasicBlock1D, num_blocks=[2, 2, 2, 2], 
                 num_classes: int = 1, input_channels: int = 20,
                 dropout_rate: float = 0.3):
        super(ResNetForPricePrediction, self).__init__()
        self.in_planes = 64
        
        # 입력 레이어 - 시계열 데이터용 1D 컨볼루션
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet 레이어들
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 글로벌 평균 풀링
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout_rate)
        
        # 완전연결층 (회귀를 위해)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 입력: (batch_size, features, sequence_length)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


def create_resnet18(num_classes: int = 1, input_channels: int = 20, 
                   dropout_rate: float = 0.3) -> ResNetForPricePrediction:
    """1D ResNet-18 모델 생성 (시계열용)"""
    return ResNetForPricePrediction(BasicBlock1D, [2, 2, 2, 2], 
                                   num_classes, input_channels, dropout_rate)


def create_resnet34(num_classes: int = 1, input_channels: int = 20,
                   dropout_rate: float = 0.3) -> ResNetForPricePrediction:
    """1D ResNet-34 모델 생성 (시계열용)"""
    return ResNetForPricePrediction(BasicBlock1D, [3, 4, 6, 3], 
                                   num_classes, input_channels, dropout_rate)