import numpy as np
import pandas as pd
from utils.metrics import metric
import os
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time 
import math
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
import itertools
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # 进度条显示

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    def __init__(self, jsonl_files, year, train=True):
        self.data = []
        self.targets = []
        for file in jsonl_files:
            # print(file)
            if 'jsonl' not in file:
                continue
            
            with open(file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    for key in entry.keys():
                        if (1980 <= int(key) < year) and train:
                            data_0 = torch.tensor(entry[key]['0'])
                            
                            self.data.append(data_0)
                            self.targets.append(torch.tensor(entry[key]['1']))
                            
                        elif (year <= int(key) <= 2007) and (train is False):                
                            data_0 = torch.tensor(entry[key]['0'])
                            
                            self.data.append(data_0)
                            self.targets.append(torch.tensor(entry[key]['1']))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def collate_fn(batch):
    features, labels = zip(*batch)
    features = tuple([item.to(device) for item in features])
    labels = tuple([item.to(device) for item in labels])
    # Pad features to the maximum sequence length in batch
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    labels = torch.stack(labels)  # Stack labels into a tensor

    masks = (features_padded.sum(dim=-1) != 0)
    # display('mask', masks.size())
    return features_padded, labels, masks
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, batch_size=64, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)  # 调整形状为 [batch_size, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x : [batch_size, seq_length, d_model]
        x = x + self.pe[:x.size(0), :x.size(1), :]  # 确保位置编码与输入张量尺寸匹配
        return self.dropout(x)

        
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, repeat_dim, num_heads, num_layers, output_dim, num_classes=13, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.repeat_dim = repeat_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding = nn.Linear(input_dim, embed_dim-repeat_dim)  # Embedding from input_dim to embed_dim
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=1024, dropout=dropout,
            activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, output_dim)  # To output a single real number

    def forward(self, src, src_mask):
        # concat_tensor = src[:, :, -1:].repeat(1,1,self.repeat_dim)
        concat_tensor = torch.repeat_interleave(src[:, :, -1:], repeats=self.repeat_dim, dim=-1)

        src = self.embedding(src[:, :, :self.input_dim])  # [batch_size, seq_length, embed_dim]

        # print(src.size())
        src = torch.concat([src, concat_tensor], dim=2)
        # print(src.size())
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_key_padding_mask=~src_mask)
        output = output.mean(dim=1)  # Pooling
        output = self.output_layer(output)
        return output


# no train loss
def no_train_loss(model, train_loader, criterion, device):
    total_loss = 0
    for features, labels, masks in train_loader:
        features, labels, masks = features.to(device), labels.to(device), masks.to(device)
        # print(features.device, masks.device)
        outputs = model(features, masks)
        loss = criterion(outputs.squeeze(), labels)
        total_loss += loss.item()
    return total_loss / len(train_loader)
    
    
# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels, masks in test_loader:
            features, labels, masks = features.to(device), labels.to(device), masks.to(device)
            outputs = model(features, masks)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)



# 训练和验证函数
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_model_wts = None
    best_val_loss = float('inf')
    best_epoch = 0  # 保存最佳 epoch


    no_train_loss_res = no_train_loss(model, train_loader, criterion, device)
    print('initial train loss: ', no_train_loss_res)
    
    no_train_loss_res = no_train_loss(model, val_loader, criterion, device)
    print('initial val loss: ', no_train_loss_res)


    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for data, labels, masks in train_loader:
            data, labels, masks = data.to(device), labels.to(device), masks.to(device)

            outputs = model(data, masks)
            loss = criterion(outputs.squeeze(), labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for data, labels, masks in val_loader:
                data, labels, masks = data.to(device), labels.to(device), masks.to(device)
                outputs = model(data, masks)
                loss = criterion(outputs.squeeze(), labels)
                
                running_val_loss += loss.item() * data.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        # 如果当前验证损失小于最佳损失，则更新最佳模型权重和最佳 epoch
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = model.state_dict()
            best_epoch = epoch + 1  # 保存最佳 epoch（从 1 开始）
        
    # 返回最佳模型权重、最佳验证损失和最佳 epoch
    return best_model_wts, best_val_loss, best_epoch
    

# 主函数，执行超参数搜索和交叉验证
def hyperparameter_search(dataset, param_grid, k_folds=5, device='cpu'):
    # 创建超参数组合
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    best_overall_loss = float('inf')
    best_params = None
    best_model_wts = None
    
    # 遍历每个超参数组合
    for param_values in tqdm(param_combinations, desc="Hyperparameter Search"):
        params = dict(zip(param_names, param_values))
        print(f"\nEvaluating parameters: {params}")
        
        # 存储每个折的验证损失
        val_losses = []
        
        # 创建KFold对象
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # 遍历每个折，记录每个折里最优的折和epoch
        record_best_val_gdp_loss = float('inf')
        record_best_epoch = 0
        record_best_fold = 0
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"Fold {fold + 1}/{k_folds}")
            
            # 创建数据加载器
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_subset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn)
            
            # 初始化模型、损失函数和优化器
            model = TransformerModel(input_dim=6144,
                                     embed_dim=params['embed_dim'],
                                     repeat_dim=params['repeat_dim'],
                                     num_heads=params['num_heads'],
                                     num_layers=params['num_layers'],
                                     num_classes=dataset.data[0].shape[0],
                                     output_dim=1,
                                     dropout=0.1).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

            # 训练和验证
            best_model_wts, best_val_gdp_loss, best_epoch = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, params['num_epochs'], device)
            print(f"Best Validation GDP Loss for fold {fold + 1}: {best_val_gdp_loss:.4f}")
            print(f"Best best_epoch {best_epoch}")
            val_losses.append(best_val_gdp_loss)

            if best_val_gdp_loss < record_best_val_gdp_loss:
                record_best_val_gdp_loss = best_val_gdp_loss
                record_best_epoch = best_epoch
                record_best_fold = fold + 1
                best_valid_model = model
                
        
        # 计算当前超参数组合的平均验证损失
        avg_val_loss = np.mean(val_losses)
        print(f"Average Validation Loss for parameters {params}: {avg_val_loss:.4f}")

        
        # 如果当前平均验证损失小于整体最佳损失，则更新最佳参数和模型权重
        if avg_val_loss < best_overall_loss:
            best_overall_loss = avg_val_loss
            best_params = params
            best_params['record_best_epoch'] = record_best_epoch
            best_params['record_best_val_gdp_loss'] = record_best_val_gdp_loss
            best_params['record_best_fold'] = record_best_fold
            model_save_path = 'checkpoints_transformer/'  + file_item.replace('.pt', '_') + 'transformer_best_valid_model.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"\nBest valid model saved to {model_save_path}")
            
    print(f"\nBest Hyperparameters: {best_params}")
    print(f"Best Average Validation Loss: {best_overall_loss:.4f}")
    return best_params, best_overall_loss


# 使用最佳超参数在整个训练数据集上训练最终模型，并计算在测试集的performance
def train_and_evaluate_final(train_dataset, test_dataset,
                             best_params, device='cpu'):
    
    # 创建TensorDataset和DataLoader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=best_params['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_fn)

    final_model = TransformerModel(input_dim=6144,
                             embed_dim=best_params['embed_dim'],
                             repeat_dim=best_params['repeat_dim'],
                             num_heads=best_params['num_heads'],
                             num_layers=best_params['num_layers'],
                             num_classes=train_dataset.data[0].shape[0],
                             output_dim=1,
                             dropout=0.1).to(device)

    final_criterion = nn.MSELoss()
    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])



    no_train_loss_res = no_train_loss(final_model, train_dataloader, final_criterion, device)
    print('initial train loss: ', no_train_loss_res)
    
    no_train_loss_res = no_train_loss(final_model, test_dataloader, final_criterion, device)
    print('initial val loss: ', no_train_loss_res)



    # 训练最终模型
    train_losses = []
    test_losses = []
    num_epochs = best_params['record_best_epoch']
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_labels, masks in train_dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            masks = masks.to(device)
            
            # 前向传播
            final_model.train()
            outputs = final_model(batch_data, masks)
            
            loss = final_criterion(outputs.squeeze(), batch_labels)
            # 反向传播和优化
            final_optimizer.zero_grad()
            loss.backward()
            final_optimizer.step()
    
            total_loss += loss.item() * batch_data.size(0)
    
        train_loss = total_loss/len(train_dataloader.dataset)
        train_losses.append(train_loss)
    
        preds = []
        trues = []
        # 评估模型
        final_model.eval()
        with torch.no_grad():
            test_loss = 0
            for batch_data, batch_targets, masks in test_dataloader:
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)
                masks = masks.to(device)
                
                outputs = final_model(batch_data, masks)  
                loss = final_criterion(outputs.squeeze(), batch_targets)
                
                test_loss += loss.item() * batch_data.size(0)    

                for item in outputs.squeeze():
                    preds.append(item.item())
                for item in batch_targets:
                    trues.append(item.item())      
            # print(len(test_dataloader.dataset))
            test_loss = test_loss / len(test_dataloader.dataset)
            test_losses.append(test_loss)
    
        preds = torch.Tensor(np.array(preds))
        trues = torch.Tensor(np.array(trues))        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    print('mae, mse, rmse, mape: ', mae, mse, rmse, mape)
    print("Training complete!")
    
    # # 保存最终模型
    model_save_path = 'checkpoints_transformer/'  + file_item.replace('.pt', '_') + 'transformer_best_final_model.pth'
    torch.save(final_model.state_dict(), model_save_path)
    print(f"\nFinal model saved to {model_save_path}")

    best_params['final model mae'] = mae
    best_params['final model mse'] = mse
    best_params['final model rmse'] = rmse
    best_params['final model mape'] = mape
    return best_params


# 使用checkpoint在测试集上测试
def eval_model(test_dataset, model_path, best_params, device='cpu'):
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    final_model = TransformerModel(input_dim=6144,
                             embed_dim=best_params['embed_dim'],
                             repeat_dim=best_params['repeat_dim'],
                             num_heads=best_params['num_heads'],
                             num_layers=best_params['num_layers'],
                             output_dim=1,
                             dropout=0.1).to(device)

    final_criterion = nn.MSELoss()
    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    
    # 3. 加载模型权重
    final_model.load_state_dict(torch.load(model_path))
    
    # 4. 设置模型为评估模式
    test_losses = []
    preds = []
    trues = []
    
    # 评估模型
    final_model.eval()
    with torch.no_grad():
        test_loss = 0
        test_gdp_loss = 0
        for batch_data, batch_targets, masks in test_dataloader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            masks = masks.to(device)
            
            outputs = final_model(batch_data, masks)
            loss = final_criterion(outputs.squeeze(), batch_targets)
            
            test_loss += loss.item() * batch_data.size(0)
            
            for item in outputs.squeeze():
                preds.append(item.item())

            for item in batch_targets:
                trues.append(item.item())     
        
        
        test_loss = test_loss / len(test_dataloader.dataset)
        test_losses.append(test_loss)

    
    preds = torch.Tensor(np.array(preds))
    trues = torch.Tensor(np.array(trues))

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

    print(f"Test Loss: {test_loss:.4f}")
    print('mae, mse, rmse, mape: ', mae, mse, rmse, mape)
    
    best_params['best val model mae'] = mae
    best_params['best val model mse'] = mse
    best_params['best val model rmse'] = rmse
    best_params['best val model mape'] = mape
    return best_params


file_item_list = ['train_dataset_mlp_13v_id_norm_6146_80-07.pth',
                  'train_dataset_mlp_13v_id_norm_6146_13-19.pth',
                  'train_dataset_mlp_13v_id_norm_6146_80-19.pth',
                  'train_dataset_light_year_14v_6146_13-19.pth',
                  'train_dataset_light_month_25v_6146_13-19.pth']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for file_item in tqdm(file_item_list[:]):
    # print(file_item)
    start_time = time.time()
    train_dataset_path = 'dataset_RT/' + file_item
    test_dataset_path = 'dataset_RT/' + file_item.replace('train_', 'test_')
    train_dataset = torch.load(train_dataset_path)
    test_dataset = torch.load(test_dataset_path)
    print('\n')
    print('train_dataset:', train_dataset_path, 'length: ', len(train_dataset))
    print('test_dataset:', test_dataset_path, 'length: ', len(test_dataset))
    
    set_seed(1)
    
    # 定义超参数网格
    param_grid = {
        'embed_dim': [512, 1024, 2048],
        'repeat_dim': [1, 100],
        'num_heads': [8],
        'num_layers': [1, 3],
        'lr': [0.001, 0.0001, 0.00001],
        'batch_size': [64],
        'num_epochs': [1000],
        'weight_decay': [0.01]
    }


    # 执行超参数搜索
    best_params, best_overall_loss = hyperparameter_search(train_dataset, param_grid, k_folds=5, device=device)
    
    best_params['best_overall_loss_average'] = best_overall_loss
    
    set_seed(1)
    best_params = train_and_evaluate_final(train_dataset, test_dataset,
                             best_params, device)
    
    
    model_path = 'checkpoints_transformer/' + file_item.replace('.pt', '_') + 'transformer_best_valid_model.pth'
    best_params = eval_model(test_dataset, model_path, best_params, device)
    best_params['train_data len'] = str(len(train_dataset.data))
    best_params['test_data len'] = str(len(test_dataset.data))
    best_params['train_targets len'] = str(len(train_dataset.targets))
    best_params['test_targets len'] = str(len(test_dataset.targets))
    pd.DataFrame([best_params]).to_csv('checkpoints_transformer/' + file_item.replace('.pt', '_') + 'best_params_res.csv')
    print('cost time: ', time.time() - start_time)
    print('\n===================Next=====================')

print('Done!')



