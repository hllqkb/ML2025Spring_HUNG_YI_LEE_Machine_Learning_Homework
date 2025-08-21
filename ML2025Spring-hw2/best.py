# 引入必要模块
# 数据操作
import pandas as pd
import numpy as np
import math
import csv
import random
import os
from monai.utils import set_determinism
from torch.utils.data import Dataset, DataLoader
# 进度条
from tqdm import tqdm
# 特征选择
from sklearn.feature_selection import SelectKBest, f_regression
# pytorch
import torch
import torch.nn as nn 
# 绘制图像
from torch.utils.tensorboard import SummaryWriter
# set random seed
def split_valid_set(data_set, valid_ratio, seed):
    np.random.seed(seed)
    indices = np.random.permutation(len(data_set))
    valid_size = int(len(data_set) * valid_ratio)
    valid_idx = indices[:valid_size]
    train_idx = indices[valid_size:]
    return data_set[train_idx], data_set[valid_idx]
def select_features(train_data,valid_data,test_data,select_all=True):
    # choose label
    y_train=train_data[:,-1]
    y_valid=valid_data[:,-1]
    # choose all features except label
    raw_x_train=train_data[:,:-1]
    raw_x_valid=valid_data[:,:-1]
    raw_x_test=test_data
    if select_all:
        feat_idx=list(range(raw_x_train.shape[1]))
        all_to_remove = {0}
        feat_idx = [x for x in feat_idx if x not in all_to_remove]
 
    else:
        # Feature selection
        feat_idx = [34, 35, 36, 43, 46, 47, 51, 52, 53,
                    54, 61, 64, 65, 69, 70, 71, 72, 79, 82, 83]
    return raw_x_train[:,feat_idx],raw_x_valid[:,feat_idx],raw_x_test[:,feat_idx],y_train,y_valid
       
class COVID19Dataset(Dataset):
    def __init__(self, features,target=None):
        if target is None:
            # predict
            self.target=target
        else:
            # train
            self.target = torch.FloatTensor(target)
        self.features = torch.FloatTensor(features)
    def __getitem__(self,idx):
        # idx means index of sample
        if self.target is None:
            return self.features[idx]
        else:
            return self.features[idx],self.target[idx]
    def __len__(self):
        return len(self.features)
# define model
class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        hidden_dims = config['layer']  # 如 [128, 64, 32]
 
        layers = []
        dims = [input_dim] + hidden_dims
 
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())  # 更先进的激活函数
 
        # 可选：加入 Dropout 或 BatchNorm 以提升泛化能力
        # layers.append(nn.Dropout(0.2))
 
        layers.append(nn.Linear(dims[-1], 1))  # 输出层：无激活
 
        self.layers = nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.layers(x)
        return x.squeeze(1)  # 输出 (B)
# Set CFG
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config={
    'layer': [128, 64, 32],
    'k':20,
    'seed': 3407,
    'select_all': False,
    'normalize': True,
    'batch_size': 128,
    'lr': 0.00035099698616997447,
    'n_epochs': 20000,
    'valid_ratio': 0.2,
    'early_stop': 400,
    'save_path': './models/model.ckpt',
    'beta1': 0.95,  # β₁
    'eps': 1e-7,  # ϵ
}
 
# train
def train(train_loader,valid_loader,model,config,device):
    criterion=nn.MSELoss()
    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['lr'],
    betas=(config['beta1'], 0.999),  # 调整 β₁
    eps=config['eps'],               # 调整 ϵ
    # weight_decay=0  # 可选：是否加入 L2 正则化
)
    writer=SummaryWriter()
    best_loss=math.inf
    early_stop_count=0
    step=0
    for epoch in range(config['n_epochs']):
        loss_record=[]
        model.train()
        train_bar=tqdm(train_loader)
        for x,y in train_bar:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad()
            outputs=model(x)
            loss=criterion(outputs,y)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())
            step+=1
        # validate
        model.eval()
        valid_loss_record=[]
        valid_bar=tqdm(valid_loader)
        for x,y in valid_bar:
            x,y=x.to(device),y.to(device)
            with torch.no_grad():
                outputs=model(x)
                loss=criterion(outputs,y)
                valid_loss_record.append(loss.detach().item())
        valid_loss=np.mean(valid_loss_record)
        writer.add_scalar('valid_loss',valid_loss,step)
        # save best model
        if valid_loss<best_loss:
            best_loss=valid_loss
            torch.save(model.state_dict(),config['save_path'])
            early_stop_count=0
            print('Saving model with loss {:.3f}...'.format(best_loss))
        else:
            early_stop_count+=1
        if early_stop_count>=config['early_stop']:
            return best_loss
    return best_loss
# set same seed
set_determinism(seed=config['seed'])
 
# 读取数据（保留 DataFrame 或 NumPy）
train_data = pd.read_csv('./ML2025Spring-hw2-public/train.csv').values
test_data = pd.read_csv('./ML2025Spring-hw2-public/test.csv').values
 
# 切分数据（不转成 Subset）
train_data, valid_data = split_valid_set(train_data, config['valid_ratio'], config['seed'])
 
# 检查标签是否有 NaN
if np.isnan(train_data[:, -1]).sum() > 0 or np.isnan(valid_data[:, -1]).sum() > 0:
    raise ValueError("Labels contain NaN values. Please check your data.")
# Normalization with Min-Max scaling 可以加一个点
if config['normalize']:
    train_min = np.min(train_data[:, 35:-1], axis=0)  # 计算每列最小值
    train_max = np.max(train_data[:, 35:-1], axis=0)  # 计算每列最大值
    # 防止除以0的情况，可以加一个很小的数
    epsilon = 1e-8
    train_range = train_max - train_min + epsilon
    # 归一化训练集
    train_data[:, 35:-1] = (train_data[:, 35:-1] - train_min) / train_range
    # 用相同的min和range归一化验证集
    valid_data[:, 35:-1] = (valid_data[:, 35:-1] - train_min) / train_range
    # 用相同的min和range归一化测试集
    test_data[:, 35:] = (test_data[:, 35:] - train_min) / train_range
x_train,x_valid,x_test,y_train,y_valid=select_features(train_data,valid_data,test_data,config['select_all'])
# convert to tensor
train_dataset=COVID19Dataset(x_train,y_train)
valid_dataset=COVID19Dataset(x_valid,y_valid)
test_dataset=COVID19Dataset(x_test)
# pin memory能保存数据到GPU的高速缓存中，能加快数据加载速度,shuffle=True表示每个epoch打乱数据
train_loader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,pin_memory=True)
valid_loader=DataLoader(valid_dataset,batch_size=config['batch_size'],shuffle=True,pin_memory=True)
test_loader=DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=False,pin_memory=True)
# create model
model=MyModel(input_dim=x_train.shape[1]).to(device)
 
# train
 
print('Training model...')
best_loss=train(train_loader,valid_loader,model,config,device)
print(f'Best validation loss: {best_loss:.3f}')
def predict(test_loader,model,device):
    model.eval()
    preds=[]
    with torch.no_grad():
        for x in tqdm(test_loader):
            x=x.to(device)
            outputs=model(x)
            # 把数据从GPU转移到CPU，才能转换为numpy数组
            preds.append(outputs.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds
# predict
predictions=predict(test_loader,model,device)
# save predictions
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
save_pred(predictions, './predictions.csv')
print('Predictions saved in ./predictions.csv')