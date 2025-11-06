import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from Preprocess import load_label
from utils import process_adj, read_mat

import pandas as pd
import warnings
import os
import sys
from scipy import io
import math
import argparse
from time import *
import torch
import torch.optim as optim
import torch.nn.functional as F

import warnings
import numpy as np

from model import SCDGN as Net
from dataset import *
from task import *
from utils import *

warnings.filterwarnings("ignore")

# 数据集配置
DATASET_CONFIG = {
    'IP': {
        'n_sp': 1100,
        'hyperspectral_path': 'HSI_datasets/Indian_pines_corrected.mat',
        'default_knn': 25
    },
    'SA': {
        'n_sp': 2700,
        'hyperspectral_path': 'HSI_datasets/Salinas_corrected.mat',
        'default_knn': 25
    },
    'PU': {
        'n_sp': 2200,
        'hyperspectral_path': 'HSI_datasets/PaviaU.mat',
        'default_knn': 25
    },
    'TT': {
        'n_sp': 2000,
        'hyperspectral_path': 'HSI_datasets/Trento.mat',
        'default_knn': 25
    },
}

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='ICML')
    parser.add_argument('--dataname', type=str, required=True, 
                      choices=['IP', 'SA', 'PU','TT'], help='Dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--train', type=int, default=1, help='Train mode')
    
    # 数据集参数
    parser.add_argument('--n_sp', type=int, required=True, help='Number of superpixels')
    parser.add_argument('--cut', type=str, default=False, help='Degree type')
    parser.add_argument('--type', type=str, default='sys', help='sys or rw')
    parser.add_argument('--knn', type=int, help='KNN graph neighbors')
    parser.add_argument('--v_input', type=int, default=1, help='T distribution freedom')
    parser.add_argument('--sigma', type=float, default=0.5, help='KNN weight parameter')
    
    # 优化器参数
    parser.add_argument('--imp_lr', type=float, default=1e-3, help='Implicit learning rate')
    parser.add_argument('--exp_lr', type=float, default=1e-5, help='Explicit learning rate')
    parser.add_argument('--imp_wd', type=float, default=1e-5, help='Implicit weight decay')
    parser.add_argument('--exp_wd', type=float, default=1e-5, help='Explicit weight decay')
    
    # 模型参数
    parser.add_argument("--hid_dim", type=int, default=512, help='Hidden dimension')
    parser.add_argument('--time', type=float, default=18.0, help='ODE end time')
    parser.add_argument('--method', type=str, default='dopri5', help='ODE solver method')
    parser.add_argument('--tol_scale', type=float, default=200.0, help='Tolerance scale')
    parser.add_argument('--add_source', type=str, default=True, help='Add source')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    
    # 损失参数
    parser.add_argument('--beta', type=float, default=1.0, help='Loss weight')
    parser.add_argument('--gamma', type=float, default=1.0, help='ICML weight')
    
    args = parser.parse_args()

    
    # 设备设置
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    args.device = device
    set_seed(args.seed)

    
    # 获取数据集配置
    # config = DATASET_CONFIG[args.dataname]
    config = DATASET_CONFIG[args.dataname].copy()
    # args.knn = args.knn if args.knn else config['default_knn']
    if args.n_sp is not None:
       config['n_sp'] = args.n_sp
    else:
       args.n_sp = config['n_sp']
    
     
 
    
    label_gt, labeled_p_in_sp = load_label(args.dataname, n_sp=config['n_sp'])
    
    # 加载CNN特征
    cnn_encoding = np.load(f'pretrained_emb/{args.dataname}_pretrained_emb/{args.dataname}_pretrained_emb.npy')
    
    # 处理超像素映射
    sp_map = read_mat(f'sp_seg_map/{args.dataname}_sp_map_{config["n_sp"]}.mat').astype(np.int32).reshape(-1)
    
    # 构建超像素特征
    N = cnn_encoding.shape[0]
    pixel_in_sp = np.zeros((N, config['n_sp']), dtype=np.float32)
    for i in range(N):
        pixel_in_sp[i, sp_map[i]] = 1
    sp_feat_np = np.matmul(pixel_in_sp.T, cnn_encoding)
    sp_feat = torch.from_numpy(sp_feat_np).to(device)
    
    # 生成邻接矩阵
    get_sp_adj_from_mat(
        sp_map_path=f'sp_seg_map/{args.dataname}_sp_map_{config["n_sp"]}.mat',
        hyperspectral_path=config['hyperspectral_path'],
        output_path=f'sp_adj/{args.dataname}_sp_adj_{config["n_sp"]}.npy'
    )
    
    # 加载并处理邻接矩阵
    adj = np.load(f'sp_adj/{args.dataname}_sp_adj_{config["n_sp"]}.npy', allow_pickle=True)
    adj = process_adj(adj).to(device)
    

    # 准备图数据
    feat = sp_feat
    in_dim = feat.shape[1]
    args.N = N = feat.shape[0]
    norm_factor, edge_index, edge_weight, adj_norm, knn, Lap = cal_norm(adj, args, feat)
    Lap_Neg = cal_Neg(adj_norm, knn, args)
    feat = feat.to(device)
    labels = label_gt
    
    # 初始化模型
    model = Net(N, edge_index, edge_weight, args).to(device)
    optimizer = optim.Adam([
        {'params': model.params_imp, 'weight_decay': args.imp_wd, 'lr': args.imp_lr},
        {'params': model.params_exp, 'weight_decay': args.exp_wd, 'lr': args.exp_lr}
    ])
    
    # 训练准备
    checkpt_file = f'./best/{args.dataname}_best.pt'
    os.makedirs(os.path.dirname(checkpt_file), exist_ok=True)
    
    
    if args.train:
        best_loss = float('inf')
        cnt_wait = 0
        EYE = torch.eye(args.N).to(device)
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            emb = model(knn, adj_norm, norm_factor)
            loss = (torch.trace(emb.T @ Lap @ emb) - 
                   args.beta * torch.trace(emb.T @ Lap_Neg @ emb) +
                   args.gamma * F.mse_loss(emb @ emb.T, EYE)) / args.N
                   
            loss.backward()
            optimizer.step()
            
            if loss < best_loss:
                best_loss = loss
                cnt_wait = 0
                torch.save(model.state_dict(), checkpt_file)
            else:
                cnt_wait += 1
                
            if cnt_wait >= args.patience or math.isnan(loss):
                print(f'\nEarly stopping at epoch {epoch}')
                break
    
    # 评估模型
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        emb = model(knn, adj_norm, norm_factor).cpu().numpy()

    
    # 聚类评估
    clustering(labels, emb, args, labeled_p_in_sp)



if __name__ == '__main__':
    main()