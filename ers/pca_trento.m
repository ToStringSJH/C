% 读取Trento数据集
load('Trento.mat');
data3D = HSI_Trento;
[M, N, B] = size(data3D);

% 数据预处理
Y_scale = scaleForSVM(reshape(data3D, M*N, B));

% 使用MATLAB内置PCA函数
p = 3;
[coeff, Y_pca] = pca(Y_scale, 'NumComponents', p);

% 重塑数据并转换为RGB图像
Y_pca_reshaped = reshape(Y_pca', p, M, N);
Y_pca_reshaped = permute(Y_pca_reshaped, [2 3 1]);
img = im2uint8(mat2gray(Y_pca_reshaped));

% 创建保存目录
if ~exist('pca_img', 'dir')
    mkdir('pca_img');
end

% 保存PNG图像
imwrite(img, 'pca_img/TT_pca.png');

% 显示结果
figure;
imshow(img);
title('Trento PCA Result');