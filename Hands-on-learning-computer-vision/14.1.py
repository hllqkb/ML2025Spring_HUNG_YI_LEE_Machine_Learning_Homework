import torch
import torch.nn as nn
import torchvision.models as models
class DeepPose(nn.Module):
    # num_keypoints=17：默认预测17个关键点 pretain=True：默认使用预训练模型权重 num_stage=3：默认使用3 级联个阶段（stage）
    def __init__(self,num_keypoints=17,pretain=True,num_stage=3):
        # super调用父类 nn.Module.__init__()，确保 DeepPose 正确初始化
        super(DeepPose,self).__init__()
        self.num_keypoints=num_keypoints
        self.num_stage=num_stage
        # 加载预训练模型ResNet50
        self.base_model=models.resnet50(pretrained=pretain)
        # 为每个级联定于回归层
        self.regression_layers=nn.ModuleList(
            self._make_regression_layers() for _ in range(num_stage)
        )
        # 每个级联阶段的最终全连接层，用于关节点预测
        self.final_layers=nn.ModuleList(
            nn.Linear(2048,num_keypoints*2) for _ in range(num_stage)
        )
    # 定义每个级联阶段的回归层
    def _make_regression_layers(self):
        # 定义回归层，由几个卷积层组成
        return nn.Sequential(
            nn.Conv2d(2048,512,kernel_size=3,padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,2048,kernel_size=1),
        )
    def forward(self,x):
        # 使用骨干网络提取特征
        features=self.backbone(x)
        keypoint_preds=torch.zeros(x.size(0),self.num_keypoints*2).to(x.device)
        # 循环每个级联阶段的回归层
        for i in range(self.num_stage):
           #将关节点预测结果与特征图拼接 
           keypoint_map=keypoint_preds.view(x.size(0),self.num_keypoints,2, 1,1) 
           keypoint_map=keypoint_map.expand(-1,-1,-1,features.size(2), features.size(3)) 
           features_with_keypoints=torch.cat([features, keypoint_map.view(x.size(0),-1,features.size(2), features.size(3))],dim=1)
           # 通过回归层进行精化
           regression_output=self.regression_layers[i](features_with_keypoints)
           #将回归输出展平并通过全连接层 
           regression_output=regression_output.view(x.size(0),-1) 
           keypoint_preds +=self.fc_layers[i](regression_output)
           #返回关节点的最终位置 
           return keypoint_preds.view(x.size(0),self.num_keypoints,2)