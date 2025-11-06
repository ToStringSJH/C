function Y_pca = my_pca(varargin)
% 修改后的PCA函数
% 输入参数：Y_scale (数据矩阵), p (主成分数，可选，默认3)

% 参数校验
if nargin < 1
    error('必须提供输入数据矩阵');
end

Y_scale = varargin{1};
p = 3; % 默认主成分数
if nargin >= 2
    p = varargin{2};
end

% 输入有效性检查
if isempty(Y_scale) || ~ismatrix(Y_scale)
    error('输入数据必须是二维矩阵');
end
[n, dim] = size(Y_scale);
if n < 2 || dim < 1
    error('输入数据矩阵需要至少2个样本和1个特征');
end
p = min(p, dim); % 确保主成分数不超过特征维度

% 手动实现PCA
mean_X = mean(Y_scale, 1);
X_centered = Y_scale - repmat(mean_X, n, 1);
cov_X = (X_centered' * X_centered) / (n - 1);
[eigenvectors, eigenvalues] = eig(cov_X);
eigenvalues = diag(eigenvalues);
[~, idx] = sort(eigenvalues, 'descend');
eigenvectors = eigenvectors(:, idx);
selected_eigenvectors = eigenvectors(:, 1:p);
Y_pca = X_centered * selected_eigenvectors;
end