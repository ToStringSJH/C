function [Y] = scaleForSVM(X)
% 对数据进行归一化处理
% 输入: X - 原始数据矩阵 [样本数 x 特征数]
% 输出: Y - 归一化后的数据矩阵

Y = zeros(size(X));
[m, n] = size(X);

for i = 1:n
    max_value = max(X(:,i));
    min_value = min(X(:,i));
    
    if max_value == min_value
        Y(:,i) = X(:,i);
    else
        Y(:,i) = (X(:,i) - min_value) / (max_value - min_value);
    end
end
end