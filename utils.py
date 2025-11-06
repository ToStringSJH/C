import torch
from sklearn.cluster import KMeans
from torch.utils.data.dataset import Dataset
import random
import numpy as np
#from cal_metric import full_metric
import scipy.sparse as sp
import scipy.io as sio

from sklearn.preprocessing import minmax_scale
from skimage.measure import regionprops
from sklearn.neighbors import kneighbors_graph
import numpy as np
import torch
import random
import torch.nn as nn

def set_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)


# def setup_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

class AEDataset(Dataset):
    def __init__(self, Datapath, transform):
        self.Datalist = np.load(Datapath)
        self.transform = transform

    def __getitem__(self, index):
        Data = self.transform(self.Datalist[index].astype('float64'))
        Data = Data.view(1, Data.shape[0], Data.shape[1], Data.shape[2])
        return Data

    def __len__(self):
        return len(self.Datalist)


def read_mat(filename):
    mat = sio.loadmat(filename)
    keys = [k for k in mat.keys() if k != '__version__' and k != '__header__' and k != '__globals__']
    arr = mat[keys[0]]
    return arr

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def process_adj(adj, norm='sym', renorm=True):
    # adj: numpy dense matrix
    adj = adj-np.diag(np.diag(adj))
    adj = sp.csr_matrix(adj)
    adj.eliminate_zeros()
    adj = sp.coo_matrix(adj)
    if renorm:
        adj = adj + sp.eye(adj.shape[0])
    # degree vector
    rowsum = np.array(adj.sum(1))
    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(
            degree_mat_inv_sqrt).tocoo()
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj).tocoo()
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
    return adj_normalized



# def create_spixel_graph(source_img, superpixel_labels, k,feat):
#     # 将高光谱图像展平为二维矩阵 [像素数×波段数]
#     #s = source_img.reshape((-1, source_img.shape[-1]))
#     # 生成关联矩阵
#     # a = create_association_mat(superpixel_labels)
#     # # print(a.shape)
#     # # print(s.shape)

#     # # 检查维度是否匹配
#     # if a.shape[0] != s.shape[0]:
#     #     raise ValueError(f"维度不匹配: a.shape={a.shape}, s.shape={s.shape}")

    
#     ss_fea = feat

#     # 构建KNN图 (使用欧氏距离)
#     adj = kneighbors_graph(ss_fea, n_neighbors=k, mode='distance', include_self=False).toarray()

#     # 自适应计算高斯核参数
#     X_var = ss_fea.var()  # 计算特征方差
#     gamma = 1.0 / (ss_fea.shape[1] * X_var) if X_var != 0 else 1.0
    
#     # 将距离转换为相似度 (高斯核)
#     adj[np.where(adj != 0)] = np.exp(-adj[np.where(adj != 0)]**2 * gamma)
    
#     np.fill_diagonal(adj, 0)  # 消除自连接

#     return adj  # 返回邻接矩阵


# def create_association_mat(superpixel_labels):
#     labels = np.unique(superpixel_labels)
#     n_labels = labels.shape[0]
#     n_pixels = superpixel_labels.shape[0] * superpixel_labels.shape[1]
#     association_mat = np.zeros((n_pixels, n_labels))
#     superpixel_labels_ = superpixel_labels.reshape(-1)
#     for i, label in enumerate(labels):
#         association_mat[np.where(label == superpixel_labels_), i] = 1
#     return association_mat


def get_sp_adj_from_mat(sp_map_path, hyperspectral_path, output_path):
    """
    生成超像素邻接矩阵 - 基于空间邻接关系
    
    当且仅当在HSI中存在至少一对相邻的像素(x_i^p, x_j^p)，且x_i^p属于第m个超像素，
    x_j^p属于第n个超像素时，a_m,n=a_n,m=1
    
    :param sp_map_path: ERS生成的超像素映射文件路径
    :param hyperspectral_path: 原始高光谱数据文件路径  
    :param output_path: 邻接矩阵保存路径
    """
    # 加载超像素映射
    sp_map = read_mat(sp_map_path).astype(np.int32)
    
    # 获取原始形状
    if len(sp_map.shape) > 2:
        height, width = sp_map.shape[:2]
    else:
        # 从高光谱数据获取形状信息
        hsi_data = read_mat(hyperspectral_path)
        if len(hsi_data.shape) == 3:
            height, width = hsi_data.shape[:2]
        else:
        
            total_pixels = sp_map.size
        
            height = width = int(np.sqrt(total_pixels))
            sp_map = sp_map.reshape(height, width)
    
    n_superpixels = np.max(sp_map) + 1
    
    adj_matrix = np.zeros((n_superpixels, n_superpixels), dtype=np.float32)
    
    # 修改为8-连通邻域，增加对角线方向
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # 遍历每个像素
    for i in range(height):
        for j in range(width):
            current_sp = sp_map[i, j]
            
            # 检查邻域像素
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                
                # 检查边界
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_sp = sp_map[ni, nj]
                    
                    # 如果邻域像素属于不同的超像素，则建立连接
                    if current_sp != neighbor_sp:
                        adj_matrix[current_sp, neighbor_sp] = 1
                        adj_matrix[neighbor_sp, current_sp] = 1
    
    
    np.fill_diagonal(adj_matrix, 0)
    
    # 自动创建目录
    import os
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存邻接矩阵
    np.save(output_path, adj_matrix)
    
    return adj_matrix

