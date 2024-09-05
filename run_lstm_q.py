import numpy as np
import pandas as pd
from utils.metrics import metric
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
import itertools
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 计算最小值和最大值, 并进行归一化
def norm_lstm_tensor(data, labels, freq='quarter'):
    if freq == 'quarter':
        drop_index = -3
    else:
        drop_index = -2
    
    # 将最后一维的数据分开
    data_to_norm = data[:, :, :drop_index]  # 除了最后一个维度
    labels_to_norm = labels[:, :drop_index]  # 除了最后一个维度

    # 计算data的最小值和最大值
    max_vals_data, _ = torch.max(data_to_norm, dim=0)  # 对第一个维度求最大值
    max_vals_data, _ = torch.max(max_vals_data, dim=0)  # 再对第二个维度求最大值
    
    min_vals_data, _ = torch.min(data_to_norm, dim=0)  # 对第一个维度求最小值
    min_vals_data, _ = torch.min(min_vals_data, dim=0)  # 再对第二个维度求最小值

    # 计算labels的最小值和最大值
    min_vals_label = labels_to_norm.min(dim=0, keepdim=True).values[-1]
    max_vals_label = labels_to_norm.max(dim=0, keepdim=True).values[-1]

    min_value_all = torch.min(min_vals_data, min_vals_label)
    max_value_all = torch.max(max_vals_data, max_vals_label)

    min_value = min_value_all[-1]
    max_value = max_value_all[-1]

    # 计算 Min-Max 归一化
    # 对 data (去除最后一个维度的部分) 进行 Min-Max 归一化
    normalized_data_to_norm = (data_to_norm - min_value_all) / (max_value_all - min_value_all)

    # 对 label (去除最后一个维度的部分) 进行 Min-Max 归一化
    normalized_labels_to_norm = (labels_to_norm - min_value_all) / (max_value_all - min_value_all)

    # cat the last dim
    normalized_data = torch.cat([normalized_data_to_norm, data[:, :, drop_index:]], dim=2)
    normalized_label = torch.cat([normalized_labels_to_norm, labels[:, drop_index:]], dim=1)
    
    return normalized_data, normalized_label, min_value, max_value


def split_lstm_dataset_by_year(data, labels, year, freq='quarter'):
    if freq == 'quarter':
        dim_index = -2
    else:
        dim_index = -1
        
    train_index_list = []
    test_index_list = []
    for i in range(len(labels)):
        if labels[i, dim_index] >= year:
            test_index_list.append(i)
        else:
            train_index_list.append(i)


    train_data = data[train_index_list, :, :dim_index-1]
    train_targets = labels[train_index_list, :dim_index-1]
    
    test_data = data[test_index_list, :, :dim_index-1]
    test_targets = labels[test_index_list, :dim_index-1]
    return train_data, test_data, train_targets, test_targets


def reverse_norm(row):
    # row = row.cpu()
    if len(row.shape) == 2:
        gap = max_value.item() - min_value.item()
        return row[:, -1] * gap + min_value.item()
    else:
        gap = max_value.item() - min_value.item()
        return row * gap + min_value.item()


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # initial hidden and cell
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_dim)
        
        # Decode the last time step of the hidden state
        out = self.fc(out[:, -1, :])
        return out


# no train loss
import time
def no_train_loss(model, train_loader, criterion, weight, device):
    model.eval()
    total_loss = 0
    total_gdp_loss = 0

    with torch.no_grad():
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward 
            outputs = model(batch_data)
            
            loss = criterion(outputs, batch_labels, weight)
            
            gdp_loss = criterion(reverse_norm(outputs),
                                 reverse_norm(batch_labels),
                                 weight)
    
            total_loss += loss.item() * batch_data.size(0)
            total_gdp_loss += gdp_loss.item() * batch_data.size(0)

    total_loss = total_loss/len(train_loader.dataset)
    total_gdp_loss = total_gdp_loss/len(train_loader.dataset)
    return total_loss, total_gdp_loss


def loss_weight(outputs, targets, weight):
    # print(outputs.shape)
    if len(outputs.shape) != 1: 
        criterion_ = nn.MSELoss(reduction='none')
        
        weights = torch.tensor([1.0] * outputs.shape[-1])
        weights[-1] = weight
        weights = weights.to(device)
    
        # Compute the element-wise MSE loss
        loss_temp = criterion_(outputs, targets)

        # Apply the weights
        weighted_loss = loss_temp * weights
        
        # Reduce the weighted loss (e.g., take the mean or sum)
        loss = weighted_loss.mean()  # or .sum()

    elif len(outputs.shape) == 1:  # GDP use MSE
        criterion_ = nn.MSELoss() 
        loss = criterion_(outputs, targets)
    else:
        raise ValueError("output shape Value Wrong!")

    return loss


# train and eval
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, weight, device):
    best_model_wts = None
    best_val_gdp_loss = float('inf')
    best_epoch = 0  # for save best epoch


    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(model, train_loader, criterion, weight, device)
    print('initial train loss: ', no_train_loss_res, 'initial gdp train loss: ', no_train_gdp_loss_res)
    
    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(model, val_loader, criterion, weight, device)
    print('initial val loss: ', no_train_loss_res, 'initial gdp val loss: ', no_train_gdp_loss_res)


    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_gdp_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, weight)
    
            gdp_loss = criterion(reverse_norm(outputs),
                                 reverse_norm(targets),
                                 weight)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_gdp_loss += gdp_loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_gdp_loss = running_gdp_loss / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        running_val_gdp_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets, weight)
                gdp_loss = criterion(reverse_norm(outputs),
                                     reverse_norm(targets),
                                     weight)
                
                running_val_loss += loss.item() * inputs.size(0)
                running_val_gdp_loss += gdp_loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_gdp_loss = running_val_gdp_loss / len(val_loader.dataset)
        
        # 如果当前验证损失小于最佳损失，则更新最佳模型权重和最佳 epoch
        if epoch_val_gdp_loss < best_val_gdp_loss:
            best_val_gdp_loss = epoch_val_gdp_loss
            best_model_wts = model.state_dict()
            best_epoch = epoch + 1  # 保存最佳 epoch（从 1 开始）
        
    # 返回最佳模型权重、最佳验证损失和最佳 epoch
    return best_model_wts, best_val_gdp_loss, best_epoch
    

# 主函数，执行超参数搜索和交叉验证
def hyperparameter_search(X, y, param_grid, k_folds=5, device='cpu'):
    # 将数据转换为TensorDataset
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    
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
            train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=params['batch_size'], shuffle=False)
            
            # 初始化模型、损失函数和优化器
            model = LSTMModel(input_dim=X.shape[-1],
                              hidden_dim=params['hidden_dim'],
                              num_layers=params['num_layers'],
                              output_dim=X.shape[-1],
                              dropout_rate=params['dropout_rate']).to(device)
            criterion = loss_weight
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

            # 训练和验证
            best_model_wts, best_val_gdp_loss, best_epoch = train_and_evaluate(model, train_loader, val_loader, 
                                                               criterion, optimizer, params['num_epochs'],
                                                                               params['weight'], device)
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
            model_save_path = 'checkpoints_lstm/'  + file_item.replace('.pt', '_') + 'lstm_best_valid_model.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"\nBest valid model saved to {model_save_path}")
            
    print(f"\nBest Hyperparameters: {best_params}")
    print(f"Best Average Validation Loss: {best_overall_loss:.4f}")
    return best_params, best_overall_loss


# 使用最佳超参数在整个训练数据集上训练最终模型，并计算在测试集的performance
def train_and_evaluate_final(train_data, test_data, train_targets, test_targets,
                             best_params, device='cpu'):
    
    # 创建TensorDataset和DataLoader
    train_dataset = TensorDataset(train_data, train_targets)
    test_dataset = TensorDataset(test_data, test_targets)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=best_params['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=best_params['batch_size'], shuffle=False)
    
    final_model = LSTMModel(input_dim=train_data.shape[-1],
                            hidden_dim=best_params['hidden_dim'],
                            num_layers=best_params['num_layers'],
                            output_dim=train_data.shape[-1],
                            dropout_rate=best_params['dropout_rate']).to(device)
    final_criterion = loss_weight
    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'],
                                  weight_decay=best_params['weight_decay'])

    weight = best_params['weight']

    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(final_model, train_dataloader, final_criterion, weight, device)
    print('initial train loss: ', no_train_loss_res, 'initial gdp train loss: ', no_train_gdp_loss_res)
    
    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(final_model, test_dataloader, final_criterion, weight, device)
    print('initial val loss: ', no_train_loss_res, 'initial gdp val loss: ', no_train_gdp_loss_res)


    # 训练最终模型
    train_losses = []
    test_losses = []
    train_gdp_losses = []
    test_gdp_losses = []
    num_epochs = best_params['record_best_epoch']
    for epoch in range(num_epochs):
        total_loss = 0
        total_gdp_loss = 0
        for batch_data, batch_labels in train_dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # 前向传播
            final_model.train()
            outputs = final_model(batch_data)
            
            loss = final_criterion(outputs, batch_labels, weight)
    
            gdp_loss = final_criterion(reverse_norm(outputs),
                                       reverse_norm(batch_labels),
                                       weight)
            # 反向传播和优化
            final_optimizer.zero_grad()
            loss.backward()
            final_optimizer.step()
    
            total_loss += loss.item() * batch_data.size(0)
            total_gdp_loss += gdp_loss.item() * batch_data.size(0)
    
        train_loss = total_loss/len(train_dataloader.dataset)
        train_gdp_loss = total_gdp_loss/len(train_dataloader.dataset)
        
        train_losses.append(train_loss)
        train_gdp_losses.append(train_gdp_loss)
    
    
        preds = []
        trues = []
        # 评估模型
        final_model.eval()
        with torch.no_grad():
            test_loss = 0
            test_gdp_loss = 0
            for batch_data, batch_targets in test_dataloader:
            
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = final_model(batch_data)
                
                loss = final_criterion(outputs, batch_targets, weight)
                
                gdp_loss = final_criterion(reverse_norm(outputs),
                                           reverse_norm(batch_targets),
                                           weight)
                
                test_loss += loss.item() * batch_data.size(0)
                test_gdp_loss += gdp_loss.item() * batch_data.size(0)
    

                outputs = reverse_norm(outputs)
                batch_targets = reverse_norm(batch_targets)

                for item in outputs:
                    preds.append(item.item())
                for item in batch_targets:
                    trues.append(item.item())      

            test_loss = test_loss / len(test_dataloader.dataset)
            test_gdp_loss = test_gdp_loss / len(test_dataloader.dataset)
            
            test_losses.append(test_loss)
            test_gdp_losses.append(test_gdp_loss)
    
        preds = torch.Tensor(np.array(preds))
        trues = torch.Tensor(np.array(trues))
        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, GDP Train Loss: {train_gdp_loss:.4f}, GDP Test Loss: {test_gdp_loss:.4f}")
    print('mae, mse, rmse, mape: ', mae, mse, rmse, mape)
    print("Training complete!")
    
    # # 保存最终模型
    model_save_path = 'checkpoints_lstm/'  + file_item.replace('.pt', '_') + 'lstm_best_final_model.pth'
    torch.save(final_model.state_dict(), model_save_path)
    print(f"\nFinal model saved to {model_save_path}")

    best_params['final model mae'] = mae
    best_params['final model mse'] = mse
    best_params['final model rmse'] = rmse
    best_params['final model mape'] = mape
    return best_params


# 使用checkpoint在测试集上测试
def eval_model(test_data, test_targets, model_path, best_params, device='cpu'):
    test_dataset = TensorDataset(test_data, test_targets)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=best_params['batch_size'], shuffle=False)
    
    final_model = LSTMModel(input_dim=test_data.shape[-1],
                            hidden_dim=best_params['hidden_dim'],
                            num_layers=best_params['num_layers'],
                            output_dim=test_data.shape[-1],
                            dropout_rate=best_params['dropout_rate']).to(device)
    final_criterion = loss_weight
    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'],
                                  weight_decay=best_params['weight_decay'])
    # 3. 加载模型权重
    final_model.load_state_dict(torch.load(model_path))
    
    # 4. 设置模型为评估模式
    test_losses = []
    test_gdp_losses = []
    preds = []
    trues = []
    weight = best_params['weight']
    # 评估模型
    final_model.eval()
    with torch.no_grad():
        test_loss = 0
        test_gdp_loss = 0
        for batch_data, batch_targets in test_dataloader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = final_model(batch_data)
            loss = final_criterion(outputs, batch_targets, weight)
            
            gdp_loss = final_criterion(reverse_norm(outputs),
                                       reverse_norm(batch_targets),
                                       weight)
            
            test_loss += loss.item() * batch_data.size(0)
            test_gdp_loss += gdp_loss.item() * batch_data.size(0)

            outputs = reverse_norm(outputs)
            batch_targets = reverse_norm(batch_targets)
            
            for item in outputs:
                preds.append(item.detach().cpu().numpy())

            for item in batch_targets:
                trues.append(item.detach().cpu().numpy())     
        
        
        test_loss = test_loss / len(test_dataloader.dataset)
        test_gdp_loss = test_gdp_loss / len(test_dataloader.dataset)
        
        test_losses.append(test_loss)
        test_gdp_losses.append(test_gdp_loss)

    
    preds = torch.Tensor(np.array(preds))
    trues = torch.Tensor(np.array(trues))

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

    print(f"Test Loss: {test_loss:.4f}, GDP Test Loss: {test_gdp_loss:.4f}")
    print('mae, mse, rmse, mape: ', mae, mse, rmse, mape)
    
    best_params['best val model mae'] = mae
    best_params['best val model mse'] = mse
    best_params['best val model rmse'] = rmse
    best_params['best val model mape'] = mape
    return best_params


file_item_list = []
for file_item in os.listdir('dataset'):
    if ('LSTM_data_' in file_item) and ('95-19' in file_item):
        file_item_list.append(file_item)
    else:
        continue

print('file item list length: ', len(file_item_list))


# select device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for file_item in tqdm(file_item_list[:]):
    print(file_item)
    start_time = time.time()
    data_path = 'dataset/' + file_item
    label_path = 'dataset/' + file_item.replace('LSTM_data', 'LSTM_label')
    
    
    set_seed(1)
    data = torch.load(data_path)
    labels = torch.load(label_path)
    
    data, labels, min_value, max_value = norm_lstm_tensor(data, labels, 'quarter')

    # 13-19 use 2019 as test dataset, other use 2018-2019 as test dataset
    if '13-19' in file_item:
        year = 2019
    else:
        year = 2018
        
    train_data, test_data, train_targets, test_targets = split_lstm_dataset_by_year(data, labels, year, freq='quarter')

    
    set_seed(1)
    
    # 定义超参数网格
    param_grid = {
        'hidden_dim': [512, 1024, 2048],
        'num_layers': [1, 2, 3],
        'dropout_rate': [0.1],
        'lr': [0.001, 0.0001, 0.00001],
        'batch_size': [64],
        'num_epochs': [1000],
        'weight': [20, 60],
        'weight_decay': [0.01]
    }


    # 执行超参数搜索
    best_params, best_overall_loss = hyperparameter_search(train_data, train_targets, param_grid, k_folds=5, device=device)
    
    best_params['best_overall_loss_average'] = best_overall_loss
    
    set_seed(1)
    best_params = train_and_evaluate_final(train_data, test_data, train_targets, test_targets,
                             best_params, device)
    
    
    model_path = 'checkpoints_lstm/' + file_item.replace('.pt', '_') + 'lstm_best_valid_model.pth'
    best_params = eval_model(test_data, test_targets, model_path, best_params, device)
    best_params['train_data shape'] = ', '.join([str(x) for x in train_data.shape])
    best_params['test_data shape'] = ', '.join([str(x) for x in test_data.shape])
    best_params['train_targets shape'] = ', '.join([str(x) for x in train_targets.shape])
    best_params['test_targets shape'] = ', '.join([str(x) for x in test_targets.shape])
    pd.DataFrame([best_params]).to_csv('checkpoints_lstm/' + file_item.replace('.pt', '_') + 'best_params_res.csv')
    print('cost time: ', time.time() - start_time)
    print('\n===================Next=====================')

print('Done!')



