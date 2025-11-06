import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
from time import *
import os
import imageio
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np
from sklearn import metrics
from munkres import Munkres
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from utils import style

from ipdb import set_trace


class GIFPloter():
    def __init__(self, ):
        self.path_list = []

    def PlotOtherLayer(self,fig,data,label,title='',fig_position0=1,fig_position1=1,fig_position2=1,s=0.1,graph=None,link=None,):
        color_list = []
        for i in range(label.shape[0]):
            color_list.append(int(label[i]))

        if data.shape[1] > 3:
            pca = PCA(n_components=2)
            data_em = pca.fit_transform(data)
        else:
            data_em = data

        # data_em = data_em-data_em.mean(axis=0)

        if data_em.shape[1] == 3:
            ax = fig.add_subplot(fig_position0, fig_position1, fig_position2, projection='3d')

            ax.scatter(data_em[:, 0], data_em[:, 1], data_em[:, 2], c=color_list, s=s, cmap='rainbow')

        if data_em.shape[1] == 2:
            ax = fig.add_subplot(fig_position0, fig_position1, fig_position2)

            if graph is not None:
                self.PlotGraph(data, graph, link)

            s = ax.scatter(data_em[:, 0], data_em[:, 1], c=label, s=s, cmap='rainbow')
            plt.axis('equal')
            if None:
                list_i_n = len(set(label.tolist()))
                # print(list_i_n)
                legend1 = ax.legend(*s.legend_elements(num=list_i_n),
                                    loc="upper left",
                                    title="Ranking")
                ax.add_artist(legend1)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.title(title)

    def AddNewFig(self,latent,label,link=None,graph=None,his_loss=None,title_='',path='./',dataset=None):
        fig = plt.figure(figsize=(5, 5))

        if latent.shape[0] <= 1000:   s=3
        elif latent.shape[0] <= 10000:   s = 1
        else:   s = 0.1

        # if latent.shape[1] <= 3:
        self.PlotOtherLayer(fig, latent, label, title=title_, fig_position0=1, fig_position1=1, fig_position2=1, graph=graph, link=link, s=s)
        plt.tight_layout()
        path_c = path + title_

        self.path_list.append(path_c)

        plt.savefig(path_c, dpi=100)
        plt.close()

    def PlotGraph(self, latent, graph, link):
        for i in range(graph.shape[0]):
            for j in range(graph.shape[0]):
                if graph[i, j] == True:
                    p1 = latent[i]
                    p2 = latent[j]
                    lik = link[i, j]
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            'gray',
                            lw=1 / lik)
                    if lik > link.min() * 1.01:
                        plt.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2,
                                str(lik)[:4],
                                fontsize=5)

    def SaveGIF(self):
        gif_images = []
        for i, path_ in enumerate(self.path_list):
            gif_images.append(imageio.imread(path_))
            if i > 0 and i < len(self.path_list)-2:
                os.remove(path_)
        imageio.mimsave(path_[:-4] + ".gif", gif_images, fps=3)

def TSNE_plot(X, label, str):
    em = TSNE(n_components=2,random_state=6).fit_transform(X)
    ploter = GIFPloter()
    ploter.AddNewFig(em, label, title_= str+".png", path='./figure/',)
    

# Clustering metrics
def spectral(W, k):
    """
    SPECTRUAL spectral clustering
    :param W: Adjacency matrix, N-by-N matrix
    :param k: number of clusters
    :return: data point cluster labels, n-by-1 vector.
    """
    w_sum = np.array(W.sum(axis=1)).reshape(-1)
    D = np.diag(w_sum)
    _D = np.diag((w_sum + np.finfo(float).eps)** (-1 / 2))
    L = D - W
    L = _D @ L @ _D
    eigval, eigvec = np.linalg.eig(L)
    eigval_argsort = eigval.real.astype(np.float32).argsort()
    F = np.take(eigvec.real.astype(np.float32), eigval_argsort[:k], axis=-1)
    idx = KMeans(n_clusters=k).fit(F).labels_
    return idx


def bestMap(L1, L2):
    # 增加空输入校验
    if len(L1) == 0 or len(L2) == 0:
        raise ValueError("输入标签不能为空")
    
    # 增加类型校验
    L1 = np.asarray(L1).astype(int)
    L2 = np.asarray(L2).astype(int)
    
    # 确保输入为一维数组
    L1 = L1.reshape(-1).astype(int)
    L2 = L2.reshape(-1).astype(int)
    
    # 获取唯一标签并检查数量
    Label1 = np.unique(L1)
    Label2 = np.unique(L2)
    nClass1 = len(Label1)
    nClass2 = len(Label2)
    
    # 标签数量对齐检查
    # if nClass1 < nClass2:
    #     raise ValueError("预测类别数(%d)小于真实类别数(%d)" % (nClass1, nClass2))
    
    # 构建代价矩阵
    G = np.zeros((nClass1, nClass2))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i,j] = np.sum((L1 == Label1[i]) & (L2 == Label2[j]))
    
    # 使用匈牙利算法匹配
    row_ind, col_ind = linear_sum_assignment(-G)
    
    # 创建映射表（处理不匹配情况）
    newL2 = np.zeros_like(L2)
    for i in range(len(col_ind)):
        if i < len(Label1) and col_ind[i] < len(Label2):
            mask = (L2 == Label2[col_ind[i]])
            newL2[mask] = Label1[row_ind[i]]
        else:
            # 处理未匹配的标签，保持原值
            pass
    
    return newL2

# def Hungarian(A):
#     _, col_ind = linear_sum_assignment(A)
#     # Cost can be found as A[row_ind, col_ind].sum()
#     return col_ind

# def bestMap(L1, L2):
#     L1 = L1.flatten(order='F').astype(float)
#     L2 = L2.flatten(order='F').astype(float)
#     if L1.size != L2.size:
#         sys.exit('size(L1) must == size(L2)')
#     Label1 = np.unique(L1)
#     nClass1 = Label1.size
#     Label2 = np.unique(L2)
#     nClass2 = Label2.size
#     nClass = max(nClass1, nClass2)

#     # For Hungarian - Label2 are Workers, Label1 are Tasks.
#     G = np.zeros([nClass, nClass]).astype(float)
#     for i in range(0, nClass2):
#         for j in range(0, nClass1):
#             G[i, j] = np.sum(np.logical_and(L2 == Label2[i], L1 == Label1[j]))

#     c = Hungarian(-G)
#     newL2 = np.zeros(L2.shape)
#     for i in range(0, nClass2):
#         newL2[L2 == Label2[i]] = Label1[c[i]]

#     # ori2mapped=[0]*nClass
#     # for i in range(0, nClass2):
#     #     newL2[L2 == Label2[i]] = Label1[c[i]]
#     #     ori_label=int(Label2[i])
#     #     mapped_label=int(Label1[c[i]])
#     #     ori2mapped[ori_label]=mapped_label
#     # ori2mapped=np.array(ori2mapped, dtype=np.int32)

#     return newL2

# def clustering_metrics(true_label, pred_label):
#     l1 = list(set(true_label))
#     numclass1 = len(l1)

#     l2 = list(set(pred_label))
#     numclass2 = len(l2)
#     if numclass1 != numclass2:
#         print('Class Not equal, Error!!!!')
#         return 0, 0, 0, 0, 0

#     cost = np.zeros((numclass1, numclass2), dtype=int)
#     for i, c1 in enumerate(l1):
#         mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
#         for j, c2 in enumerate(l2):
#             mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
#             cost[i][j] = len(mps_d)

#     # match two clustering results by Munkres algorithm
#     m = Munkres()
#     cost = cost.__neg__().tolist()

#     indexes = m.compute(cost)
#     idx = indexes[2][1]
#     # get the match results
#     new_predict = np.zeros(len(pred_label))
#     for i, c in enumerate(l1):
#         # correponding label in l2:
#         c2 = l2[indexes[i][1]]
#         # ai is the index with label==c2 in the predict list
#         ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
#         new_predict[ai] = c

#     acc = metrics.accuracy_score(true_label, new_predict)
#     f1_macro = metrics.f1_score(true_label, new_predict, average='macro')
#     nmi = metrics.normalized_mutual_info_score(true_label, pred_label)
#     ari = metrics.adjusted_rand_score(true_label, pred_label)

#     return acc* 100, f1_macro* 100, nmi* 100, ari* 100, idx

def clustering_metrics(true_label, pred_label):
    l1 = list(set(true_label))
    numclass1 = len(l1)

    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        return 0, 0, 0, 0, 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)
    idx = indexes[2][1]
    # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]
        # ai is the index with label==c2 in the predict list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(true_label, new_predict)
    
    # 计算Kappa系数
    kappa = metrics.cohen_kappa_score(true_label, new_predict)
    
    nmi = metrics.normalized_mutual_info_score(true_label, pred_label)
    ari = metrics.adjusted_rand_score(true_label, pred_label)

    return acc* 100, kappa* 100, nmi* 100, ari* 100, idx

def clustering(labels,embeds,args,labeled_p_in_sp):
    # labels = torch.from_numpy(labels).type(torch.LongTensor)
    #num_classes = np.max(labels).item()
    num_classes = len(np.unique(labels))
    
    accs = []
    kappas = []  
    nmis = []
    aris = []
    
    for i in range(10):
        best_acc = 0
        best_kappa = 0  
        best_nmi = 0
        best_ari = 0
        
        for j in range(10):
            predY = KMeans(n_clusters=num_classes).fit(embeds).labels_
            
            # 分析完整预测结果
            unique_pred, counts_pred = np.unique(predY, return_counts=True)
            nY=np.unique(predY, return_counts=True)
            
            pixel_pred = predY[labeled_p_in_sp]
            unique_sampled, counts_sampled = np.unique(pixel_pred, return_counts=True)
            gnd_Y = bestMap(pixel_pred,labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)
            
            acc_temp, kappa_temp, nmi_temp, ari_temp, _ = clustering_metrics(gnd_Y,pixel_pred) 
            if acc_temp > best_acc:
                best_acc = acc_temp
                best_kappa = kappa_temp
                best_nmi = nmi_temp
                best_ari = ari_temp
             
        accs.append(best_acc)
        kappas.append(best_kappa)  
        nmis.append(best_nmi)
        aris.append(best_ari)
        
    accs = np.stack(accs)
    kappas = np.stack(kappas)  
    nmis = np.stack(nmis)
    aris = np.stack(aris)
    
    print(accs)
    print(style.YELLOW + '\nClustering result: ACC:{:.2f}'.format(accs.mean().item()),'$\pm$','{:.2f}'.format(accs.std().item()),\
                                            'Kappa:{:.2f}'.format(kappas.mean().item()),'$\pm$','{:.2f}'.format(kappas.std().item()),\
                                            'NMI:{:.2f}'.format(nmis.mean().item()),'$\pm$','{:.2f}'.format(nmis.std().item()),\
                                            'ARI:{:.2f}'.format(aris.mean().item()),'$\pm$','{:.2f}'.format(aris.std().item()),)
    
    return pixel_pred



