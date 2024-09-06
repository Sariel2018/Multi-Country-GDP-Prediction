import pandas as pd
import argparse
import torch
from torch import nn, optim
from tqdm import tqdm
from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from utils.metrics import metric
import itertools
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from utils.tools import del_files

os.environ["CUDA_VISIBLE_DEVICES"]="0"


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

    # 重新拼接保留的最后一个维度
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
    if len(row.shape) == 2:
        gap = max_value.item() - min_value.item()
        return row[:, -1] * gap + min_value.item()
    else:
        gap = max_value.item() - min_value.item()
        return row * gap + min_value.item()


def get_renew_data(data):
    gdp_index = data.shape[-1] - 1
    
    data_renew = []
    gdp_data_rewnew = []
    for x in range(data.shape[0]):
        for y in range(data.shape[2]):
            renew_item = data[x][:, y:y+1]
            # for item in renew_item:
            data_renew.append(renew_item.numpy())
            if y == gdp_index:
                gdp_data_rewnew.append(renew_item.numpy())

    return torch.Tensor(data_renew), torch.Tensor(gdp_data_rewnew)


# no train loss
import time
def no_train_loss(model, train_loader, criterion, args, device):
    train_loss = []
    gdp_train_loss = []
    iter_count = 0
    model.eval()
    epoch_time = time.time()

    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = torch.Tensor(np.zeros(2))
            batch_y_mark = torch.Tensor(np.zeros(2))
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
    
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            
            loss = criterion(outputs, batch_y)
            gdp_loss = criterion(reverse_norm(outputs),
                                 reverse_norm(batch_y))
            
            train_loss.append(loss.item())
            gdp_train_loss.append(gdp_loss.item())
            
    return np.average(train_loss), np.average(gdp_train_loss)


def vali(args, model, vali_loader, criterion, mae_metric, device):
    total_loss = 0
    total_mae_loss = 0
    preds = []
    trues = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            
            batch_x_mark = torch.Tensor(np.zeros(2))
            batch_y_mark = torch.Tensor(np.zeros(2))

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(reverse_norm(pred),
                             reverse_norm(true))

            mae_loss = mae_metric(reverse_norm(pred),
                                  reverse_norm(true))

            outputs = reverse_norm(outputs)
            batch_y = reverse_norm(batch_y)
            
            for item in outputs:
                preds.append(item.detach().cpu().numpy())

            for item in batch_y:
                trues.append(item.detach().cpu().numpy())  

            total_loss += loss.item() * batch_x.size(0)
            total_mae_loss += mae_loss.item() * batch_x.size(0)

    
    preds = torch.Tensor(np.array(preds))
    trues = torch.Tensor(np.array(trues))

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mae, mse, rmse, mape: ', mae, mse, rmse, mape)

    total_loss = total_loss / len(vali_loader.dataset)
    total_mae_loss = total_mae_loss / len(vali_loader.dataset)

    model.train()
    return total_loss, total_mae_loss, mae, mse, rmse


# 训练和验证函数
def train_and_evaluate(model, train_loader, gdp_val_loader, criterion, model_optim, num_epochs, args, device):
    best_model_wts = None
    best_val_gdp_loss = float('inf')
    best_epoch = 0  # 保存最佳 epoch
    mae_metric = nn.L1Loss()

    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(model, train_loader, criterion, args, device)
    print('initial train loss: ', no_train_loss_res, 'initial gdp train loss: ', no_train_gdp_loss_res)
    
    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(model, gdp_val_loader, criterion, args, device)
    print('initial val loss: ', no_train_loss_res, 'initial gdp val loss: ', no_train_gdp_loss_res)

    # val_gdp_losses = []
    for epoch in tqdm(range(num_epochs)):
        iter_count = 0
        train_loss = 0
    
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
    
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = torch.Tensor(np.zeros(2))
            batch_y_mark = torch.Tensor(np.zeros(2))
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
    
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                device)
    
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
            # train_loss.append(loss.item())
            train_loss += loss.item() * batch_x.size(0)
    
            loss.backward()
            model_optim.step()
    
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = train_loss / len(train_loader.dataset)

        val_gdp_loss, val_gdp_mae_loss, mae, mse, rmse = vali(args, model,
                                              gdp_val_loader,
                                              criterion, mae_metric, device)
        
        print( #         Test Loss: {3:.7f} MAE Loss: {4:.7f} \
            "Epoch: {0} | Train Loss: {1:.7f} Vali GDP Loss: {2:.7f} \
            MAE GDP Loss: {3:.7f}".format(
                epoch + 1, train_loss,
                val_gdp_loss, val_gdp_mae_loss))
    
        epoch_val_gdp_loss = val_gdp_loss
    
        # 如果当前验证损失小于最佳损失，则更新最佳模型权重和最佳 epoch
        if epoch_val_gdp_loss < best_val_gdp_loss:
            best_val_gdp_loss = epoch_val_gdp_loss
            best_model_wts = model.state_dict()
            best_epoch = epoch + 1  # 保存最佳 epoch（从 1 开始）
        
    # 返回最佳模型权重、最佳验证损失和最佳 epoch
    return best_model_wts, best_val_gdp_loss, best_epoch


# 主函数，执行超参数搜索和交叉验证
def hyperparameter_search(X, y, param_grid, k_folds=5, device='cuda'):
    args = argparse.Namespace(
                          task_name='short_term_forecast', 
                          model_id='test', 
                          model='TimeLLM', 
                          data='GDP', 
                          features='M', 
                          seq_len=X.shape[1], 
                          label_len=0, 
                          pred_len=1, 

                          enc_in=7, 
                          dec_in=7, 
                          c_out=7, 
                          d_model=32, 
                          n_heads=8, 
                          e_layers=2, 
                          d_layers=1, 
                          d_ff=128, 
                          moving_avg=25, 
                          factor=1, 
                          dropout=0.1, 
                          embed='timeF', 
                          activation='gelu', 
                          output_attention=False, 
                          patch_len=5, 
                          stride=1, 
                          prompt_domain=0, 
                          llm_model='GPT2',
                          llm_dim=768,  # GPT2 768, LLAMA 4096
                          num_workers=20, 
                          itr=1,
                          train_epochs=1000,
                          align_epochs=10, 
                          batch_size=64,
                          eval_batch_size=64, 
                          patience=1000, 
                          learning_rate=0.001, 
                          des='test',
                          loss='MSE', 
                          lradj='type1',
                          pct_start=0.2,
                          use_amp=False, 
                          llm_layers=6,
                          percent=100)
    
    
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

        # update params
        for key, value in params.items():
            vars(args)[key] = value

        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des, 1)
    
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
    
            train_subset_data, gdp_train_subset_data = get_renew_data(train_subset[:][0])
            train_subset_label, gdp_train_subset_label = get_renew_data(train_subset[:][1])

            val_subset_data, gdp_val_subset_data = get_renew_data(val_subset[:][0])
            val_subset_label, gdp_val_subset_label = get_renew_data(val_subset[:][1])
            
            train_subset = TensorDataset(torch.tensor(train_subset_data, dtype=torch.float32),
                                         torch.tensor(train_subset_label, dtype=torch.float32))
            gdp_train_subset = TensorDataset(torch.tensor(gdp_train_subset_data, dtype=torch.float32),
                                         torch.tensor(gdp_train_subset_label, dtype=torch.float32))
            
            val_subset = TensorDataset(torch.tensor(val_subset_data, dtype=torch.float32),
                                         torch.tensor(val_subset_label, dtype=torch.float32))
            gdp_val_subset = TensorDataset(torch.tensor(gdp_val_subset_data, dtype=torch.float32),
                                         torch.tensor(gdp_val_subset_label, dtype=torch.float32))
            
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False)

            gdp_train_loader = DataLoader(gdp_train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False)
            gdp_val_loader = DataLoader(gdp_val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False)

            print("Loader Done!")
            
            # 初始化模型、损失函数和优化器
            if args.model == 'Autoformer':
                model = Autoformer.Model(args).float()
            elif args.model == 'DLinear':
                model = DLinear.Model(args).float()
            else:
                model = TimeLLM.Model(args).float()
                
            model = model.to(device)
            
            time_now = time.time()
            
            train_steps = len(train_loader)
            
            model_optim = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
            
            criterion = nn.MSELoss()
            mae_metric = nn.L1Loss()

            # 训练和验证
            best_model_wts, best_val_gdp_loss, best_epoch = train_and_evaluate(model, train_loader, gdp_val_loader, 
                                                               criterion, model_optim, params['train_epochs'],
                                                                               args, device)
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
            model_save_path = 'checkpoints_timellm/'  + file_item.replace('.pt', '_') + 'timellm_best_valid_model.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"\nBest valid model saved to {model_save_path}")
            
    print(f"\nBest Hyperparameters: {best_params}")
    print(f"Best Average Validation Loss: {best_overall_loss:.4f}")
    return best_params, best_overall_loss


# 使用最佳超参数在整个训练数据集上训练最终模型，并计算在测试集的performance
def train_and_evaluate_final(train_data, test_data, train_targets, test_targets,
                             best_params, device='cpu'):

    args = argparse.Namespace(
                          task_name='short_term_forecast', 
                          model_id='test', 
                          model='TimeLLM', 
                          data='GDP', 
                          features='M', 
                          seq_len=train_data.shape[1], 
                          label_len=0, 
                          pred_len=1, 
        
                          enc_in=7, 
                          dec_in=7, 
                          c_out=7, 
                          d_model=32, 
                          n_heads=8, 
                          e_layers=2, 
                          d_layers=1, 
                          d_ff=128, 
                          moving_avg=25, 
                          factor=1, 
                          dropout=0.1, 
                          embed='timeF', 
                          activation='gelu', 
                          output_attention=False, 
                          patch_len=5, 
                          stride=1, 
                          prompt_domain=0, 
                          llm_model='GPT2',
                          llm_dim=768,  # GPT2 768, LLAMA 4096
                          num_workers=20, 
                          itr=1,
                          train_epochs=1000,
                          align_epochs=10, 
                          batch_size=64,
                          eval_batch_size=64, 
                          patience=1000, 
                          learning_rate=0.001, 
                          des='test',
                          loss='MSE', 
                          lradj='type1',
                          pct_start=0.2,
                          use_amp=False, 
                          llm_layers=6,
                          percent=100)
    
    # update params
    for key, value in best_params.items():
        vars(args)[key] = value
    
    # 将数据转换为TensorDataset
    train_data, gdp_train_data = get_renew_data(train_data)
    train_targets, gdp_train_targets = get_renew_data(train_targets)

    test_data, gdp_test_data = get_renew_data(test_data)
    test_targets, gdp_test_targets = get_renew_data(test_targets)

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                                 torch.tensor(train_targets, dtype=torch.float32))
    gdp_train_dataset = TensorDataset(torch.tensor(gdp_train_data, dtype=torch.float32),
                                 torch.tensor(gdp_train_targets, dtype=torch.float32))
    
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32),
                                 torch.tensor(test_targets, dtype=torch.float32))
    gdp_test_dataset = TensorDataset(torch.tensor(gdp_test_data, dtype=torch.float32),
                                 torch.tensor(gdp_test_targets, dtype=torch.float32))

          
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    gdp_train_loader = DataLoader(gdp_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    gdp_test_loader = DataLoader(gdp_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print("Loader Done!")
    
    # 初始化模型、损失函数和优化器
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()
    model = model.to(device)

    time_now = time.time()
    train_steps = len(train_loader)
    model_optim = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    # 训练和验证
    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(model, train_loader, criterion, args, device)
    print('initial train loss: ', no_train_loss_res, 'initial gdp train loss: ', no_train_gdp_loss_res)
    
    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(model, gdp_test_loader, criterion, args, device)
    print('initial val loss: ', no_train_loss_res, 'initial gdp val loss: ', no_train_gdp_loss_res)

    num_epochs = best_params['record_best_epoch']
    for epoch in tqdm(range(num_epochs)):
        iter_count = 0
        train_loss = 0
    
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
    
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = torch.Tensor(np.zeros(2))
            batch_y_mark = torch.Tensor(np.zeros(2))
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
    
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                device)
    
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
            # train_loss.append(loss.item())
            train_loss += loss.item() * batch_x.size(0)

            loss.backward()
            model_optim.step()
    
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        # train_loss = np.average(train_loss)
        train_loss = train_loss / len(train_loader.dataset)

        val_gdp_loss, val_gdp_mae_loss, mae, mse, rmse = vali(args, model,
                                              gdp_test_loader,
                                              criterion, mae_metric, device)
        
        print( #         Test Loss: {3:.7f} MAE Loss: {4:.7f} \
            "Epoch: {0} | Train Loss: {1:.7f} Vali GDP Loss: {2:.7f} \
            MAE GDP Loss: {3:.7f}".format(
                epoch + 1, train_loss,
                val_gdp_loss, val_gdp_mae_loss))

    print("Complete!")
    
    # # 保存最终模型
    model_save_path = 'checkpoints_timellm/'  + file_item.replace('.pt', '_') + 'timellm_best_final_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\nFinal model saved to {model_save_path}")

    best_params['final model mae'] = mae
    best_params['final model mse'] = mse
    best_params['final model rmse'] = rmse
    return best_params


# 使用checkpoint在测试集上测试
def eval_model(test_data, test_targets, model_path, best_params, device='cpu'):

    args = argparse.Namespace(
                          task_name='short_term_forecast', 
                          model_id='test', 
                          model='TimeLLM', 
                          data='GDP', 
                          features='M', 
                          seq_len=test_data.shape[1], 
                          label_len=0, 
                          pred_len=1, 

                          enc_in=7, 
                          dec_in=7, 
                          c_out=7, 
                          d_model=32, 
                          n_heads=8, 
                          e_layers=2, 
                          d_layers=1, 
                          d_ff=128, 
                          moving_avg=25, 
                          factor=1, 
                          dropout=0.1, 
                          embed='timeF', 
                          activation='gelu', 
                          output_attention=False, 
                          patch_len=5, 
                          stride=1, 
                          prompt_domain=0, 
                          llm_model='GPT2',
                          llm_dim=768,  # GPT2 768, LLAMA 4096
                          num_workers=20, 
                          itr=1,
                          train_epochs=1000,
                          align_epochs=10, 
                          batch_size=64,
                          eval_batch_size=64, 
                          patience=1000, 
                          learning_rate=0.001, 
                          des='test',
                          loss='MSE', 
                          lradj='type1',
                          pct_start=0.2,
                          use_amp=False, 
                          llm_layers=6,
                          percent=100)
    
    # update params
    for key, value in best_params.items():
        vars(args)[key] = value
    
    
    # 将数据转换为TensorDataset
    test_data, gdp_test_data = get_renew_data(test_data)
    test_targets, gdp_test_targets = get_renew_data(test_targets)

    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32),
                                 torch.tensor(test_targets, dtype=torch.float32))
    gdp_test_dataset = TensorDataset(torch.tensor(gdp_test_data, dtype=torch.float32),
                                 torch.tensor(gdp_test_targets, dtype=torch.float32))
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    gdp_test_loader = DataLoader(gdp_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print("Loader Done!")
    
    # 初始化模型、损失函数和优化器
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()
    model = model.to(device)
    
    time_now = time.time()
    model_optim = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    
    # 3. 加载模型权重
    model.load_state_dict(torch.load(model_path))

    
    # 评估模型
    val_gdp_loss, val_gdp_mae_loss, mae, mse, rmse = vali(args, model,
                                          gdp_test_loader,
                                          criterion, mae_metric, device)
    
    print( #         Test Loss: {3:.7f} MAE Loss: {4:.7f} \
        "Vali GDP Loss: {0:.7f} \
        MAE GDP Loss: {1:.7f}".format(
            val_gdp_loss, val_gdp_mae_loss))


    print(f"GDP MSE Test Loss: {val_gdp_loss:.4f}, GDP MAE Test Loss: {val_gdp_mae_loss:.4f}")
    print('mae, mse, rmse, mape: ', mae, mse, rmse)
    
    best_params['best val model mae'] = mae
    best_params['best val model mse'] = mse
    best_params['best val model rmse'] = rmse
    return best_params



file_item_list = []
for file_item in os.listdir('../dataset/'):
    if ('LSTM_data_' in file_item):
        file_item_list.append(file_item)
    else:
        continue

print('file item list length: ', len(file_item_list))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for file_item in tqdm(file_item_list[:]):
    print(file_item)
    start_time = time.time()

    set_seed(1)
    
    data_path = '../dataset/' + file_item
    label_path = '../dataset/' + file_item.replace('LSTM_data', 'LSTM_label')
    
    
    data = torch.load(data_path)
    labels = torch.load(label_path)
    
    data, labels, min_value, max_value = norm_lstm_tensor(data, labels, 'quarter')
    
    if '13-19' in file_item:
        year = 2019
    else:
        year = 2018
        
    train_data, test_data, train_targets, test_targets = split_lstm_dataset_by_year(data, labels, year, freq='quarter')
    
    train_targets = train_targets.unsqueeze(1)
    test_targets = test_targets.unsqueeze(1)

    print('train_data shape ', train_data.shape)
    print('test_data shape ', test_data.shape)

    
    set_seed(1)
    
    param_grid = {
        'd_model': [32, 128, 512],
        'learning_rate': [0.01, 0.001],
        'train_epochs': [20],
        'patch_len': [3, 5],
        'description_add': ""
    }
    
    if ('_data_q_' in file_item) or ('_data_light_' in file_item):
        description_add = "Gross Domestic Product (GDP) is a measure of the total monetary value of all finished goods and services produced within a country’s borders in a specific time period. This dataset consists some economic indicators, such as 'Export Value', 'Industrial Added Value', 'Stock Market Capitalization', 'Balance of Payments - Financial Account Balance', 'Net International Investment Position', 'Import Value', 'Nominal Effective Exchange Rate', 'Retail Sales', 'CPI (Consumer Price Index)', 'Unemployment Rate' and 'Central Bank Policy Rate', etc."
    elif ('_data_gdp_q_' in file_item) or ('_data_gdp_more_q_' in file_item):
        description_add = "Gross Domestic Product (GDP) is a measure of the total monetary value of all finished goods and services produced within a country’s borders in a specific time period. This dataset consists only GDP."
    elif ('_gdp_more_light_' in file_item) or ('_gdp_light_' in file_item):
        description_add = "Gross Domestic Product (GDP) is a measure of the total monetary value of all finished goods and services produced within a country’s borders in a specific time period. This dataset consists GDP and some Nighttime light remote sensing data, which refers to the use of remote sensing technology to capture the distribution of lights on Earth at night."

    param_grid['description_add'] = [description_add]
    
    # 选择设备（如果有GPU可用，则使用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 执行超参数搜索
    best_params, best_overall_loss = hyperparameter_search(train_data, train_targets, param_grid, k_folds=5, device=device)
    
    best_params['best_overall_loss_average'] = best_overall_loss
    
    set_seed(1)
    best_params = train_and_evaluate_final(train_data, test_data, train_targets, test_targets,
                             best_params, device)
    
    model_path = 'checkpoints_timellm/' + file_item.replace('.pt', '_') + 'timellm_best_valid_model.pth'
    best_params = eval_model(test_data, test_targets, model_path, best_params, device)
    best_params['train_data shape'] = ', '.join([str(x) for x in train_data.shape])
    best_params['test_data shape'] = ', '.join([str(x) for x in test_data.shape])
    best_params['train_targets shape'] = ', '.join([str(x) for x in train_targets.shape])
    best_params['test_targets shape'] = ', '.join([str(x) for x in test_targets.shape])
    pd.DataFrame([best_params]).to_csv('checkpoints_timellm/' + file_item.replace('.pt', '_') + 'best_params_res.csv')
    print('cost time: ', time.time() - start_time)
    print('\n===================Next=====================')
    
print('Done!')


