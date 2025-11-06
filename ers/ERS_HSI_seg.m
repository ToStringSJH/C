% close all;clear;clc;
% disp('Entropy Rate Superpixel Segmentation Demo');
% 
% dataset='IP';
% file_path=['pca_img/', dataset, '_pca.png'];
% img = imread(file_path);
% grey_img = double(rgb2gray(img));
% 
% n_sp=2200;
% 
% seg_res = mex_ers(grey_img, n_sp);
% seg_res = int32(seg_res);
% save_file_path=['seg_res/', dataset, '/', dataset, '_sp_map_', num2str(n_sp), '.mat'];
% save(save_file_path,'seg_res');



% disp('Entropy Rate Superpixel Segmentation Demo');
% 
% dataset='SA';  
% file_path=fullfile('pca_img', [dataset '_pca.png']);
% 
% % ==== 调试信息 ====
% fprintf('\n=== 开始执行，当前数据集：%s ===\n', dataset);
% fprintf('1. 正在检查PCA图像文件：%s\n', file_path);
% if ~exist(file_path, 'file')
%     error('PCA图像文件不存在，请先运行PCA预处理');
% else
%     fprintf('√ 文件存在\n');
% end
% % ====================
% 
% % 目录创建部分保持不变...
% 
% % 新增图像加载验证
% fprintf('2. 正在加载图像...\n');
% img = imread(file_path);
% fprintf('√ 图像尺寸：%dx%d\n', size(img,1), size(img,2));
% 
% % 新增灰度转换验证
% fprintf('3. 转换为灰度图像...\n');
% grey_img = double(rgb2gray(img));
% fprintf('√ 灰度矩阵范围：[%.2f, %.2f]\n', min(grey_img(:)), max(grey_img(:)));
% 
% % 超像素分割验证
% fprintf('4. 开始超像素分割(n_sp=%d)...\n', n_sp);
% seg_res = mex_ers(grey_img, n_sp);
% fprintf('√ 分割完成，结果尺寸：%dx%d\n', size(seg_res));
% 
% % 结果保存验证
% fprintf('5. 正在保存结果至：%s\n', save_file_path);
% save(save_file_path,'seg_res');
% if exist(save_file_path, 'file')
%     fprintf('√ 保存成功\n');
% else
%     error('× 保存失败');
% end


% close all;clear;clc;
% disp('Entropy Rate Superpixel Segmentation Demo');
% 
% dataset = 'IP';
% file_path = fullfile('pca_img', [dataset, '_pca.png']);
% img = imread(file_path);
% grey_img = double(rgb2gray(img));
% 
% n_sp = 5500;
% 
% seg_res = mex_ers(grey_img, n_sp);
% seg_res = int32(seg_res);
% 
% % 创建保存目录
% save_dir = fullfile('seg_res', dataset);
% if ~exist(save_dir, 'dir')
%     mkdir(save_dir);  % 创建多级目录
%     fprintf('已创建目录：%s\n', save_dir);
% end
% 
% % 构建完整保存路径
% save_file_path = fullfile(save_dir, [dataset, '_sp_map_', num2str(n_sp), '.mat']);
% 
% % 添加错误处理
% try
%     save(save_file_path, 'seg_res');
%     fprintf('成功保存至：%s\n', save_file_path);
% catch ME
%     error('保存失败！原因：%s\n检查：\n1. 磁盘空间\n2. 写权限\n3. 文件是否被占用', ME.message);
% end

close all;clear;clc;
disp('Entropy Rate Superpixel Batch Generation');

% 数据集配置
datasets = {
    %struct('name','IP', 'nsp_range',1100:500:3600),   % 1100,1600,...,3600
    %struct('name','SA', 'nsp_range',2700:500:5200),   % 2700,3200,...,5200
    %struct('name','PU', 'nsp_range',5000:3000:20000)    % 2200,2700,...,4700
 %struct('name','TT', 'nsp_range',5000:3000:20000) 
 struct('name','TT', 'nsp_range',3000:500:5500) 
};

for d = 1:length(datasets)
    dataset = datasets{d}.name;
    fprintf('\nProcessing %s...\n', dataset);
    
    % 加载PCA图像
    img_path = fullfile('pca_img', [dataset '_pca.png']);
    if ~exist(img_path, 'file')
        error('PCA图像不存在: %s', img_path);
    end
    img = imread(img_path);
    grey_img = double(rgb2gray(img));
    
    % 遍历所有n_sp参数
    for n_sp = datasets{d}.nsp_range
        fprintf('Generating n_sp=%d...', n_sp);
        
        % 执行超像素分割
        seg_res = mex_ers(grey_img, n_sp);
        seg_res = int32(seg_res);
        
        % 创建保存目录
        save_dir = fullfile('seg_res', dataset);
        if ~exist(save_dir, 'dir')
            mkdir(save_dir);
        end
        
        % 保存结果
        save_path = fullfile(save_dir, sprintf('%s_sp_map_%d.mat', dataset, n_sp));
        try
            save(save_path, 'seg_res');
            fprintf('保存成功\n');
        catch ME
            fprintf('保存失败: %s\n', ME.message);
        end
    end
end
disp('所有参数组合处理完成！');