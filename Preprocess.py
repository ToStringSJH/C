import numpy as np
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from operator import truediv
from utils import read_mat
from utils import AEDataset, latent_loss,get_sp_adj_from_mat

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, windowSize=5):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchIndex = patchIndex + 1
    return patchesData

def loadData(name):
    data_path = 'HSI_datasets'
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
    return data

# def feature_normalize(data):
#     mu = np.mean(data,axis=0)
#     std = np.std(data,axis=0)
#     return truediv((data - mu),std)

def feature_normalize(data):
    """
    :param data: 原始数据数组
    """
    # 分块计算统计量
    chunk_size = 1000
    sum_x = np.zeros(data.shape[1:], dtype=np.float32)
    sum_x2 = np.zeros(data.shape[1:], dtype=np.float32)
    
    for i in range(0, data.shape[0], chunk_size):
        chunk = data[i:i+chunk_size].astype(np.float32)
        sum_x += np.sum(chunk, axis=0)
        sum_x2 += np.sum(chunk**2, axis=0)
    
    mu = sum_x / data.shape[0]
    std = np.sqrt((sum_x2/data.shape[0]) - mu**2)
    std[std < 1e-8] = 1.0

    # 分块归一化
    normalized = np.empty(data.shape, dtype=np.float32)
    for i in range(0, data.shape[0], chunk_size):
        chunk = data[i:i+chunk_size].astype(np.float32)
        normalized[i:i+chunk_size] = (chunk - mu) / std
    
    return normalized

def Preprocess(XPath, dataset, Windowsize=25, Patch_channel=15):
    X = loadData(dataset)
    X = applyPCA(X, numComponents=Patch_channel)
    X = createImageCubes(X, windowSize=Windowsize)

    output_dir = os.path.dirname(XPath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"已创建输出目录：{output_dir}")
    X = feature_normalize(X)
    np.save(XPath, X)


def load_label(dataset_name, n_sp=1000):
    if dataset_name=='IP':
        label_path='HSI_datasets/Indian_pines_gt.mat'
    elif dataset_name=='SA':
        label_path='HSI_datasets/Salinas_gt.mat'
    elif dataset_name=='PU':
        label_path='HSI_datasets/PaviaU_gt.mat'
    elif dataset_name=='TT':
        label_path='HSI_datasets/Trento_gt.mat'

    label=read_mat(label_path).astype(np.int32)-1

    label=label.reshape(-1)
    labeled_ids=np.where(label>=0)[0]
    label_gt=label[labeled_ids]

    #read sp seg res
    sp_map=read_mat(f'sp_seg_map/{dataset_name}_sp_map_{n_sp}.mat').astype(np.int32)

    #labeled pixel in which superpixel
    sp_map=sp_map.reshape(-1)
    labeled_p_in_sp=sp_map[labeled_ids]
    # unique_gt=np.unique(label_gt)
    # print('unique_gt:',len(unique_gt))
    # unique_in=np.unique(labeled_p_in_sp)
    # print('unique_in:',len(unique_in))

    return label_gt, labeled_p_in_sp