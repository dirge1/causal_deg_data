% clear; clc; close all;

file_prefixes = {
    'result_pc_data_numerical_var3_vec',...
    'result_ges_data_numerical_var3_vec',...
    'result_LINGAM_direct_data_numerical_var3_vec', ...
    'result_notears_linear_data_numerical_var3_vec', ...
    'result_notears_mlp_data_numerical_var3_vec', ...
    'result_CaPS_data_numerical_var3_vec'  
};

method_names = {'Stable-PC', 'GES', 'Direct-LiNGAM', ...
                'NOTEARS-Linear', 'NOTEARS-MLP', 'CaPS'};

sample_sizes = 1:10; % 样本量
num_methods = length(file_prefixes);
num_runs = 20; % 每个样本的运行次数
fosize = 14; % 字体大小
correct_matrix = [0 0; 0 0]; % 期望的正确矩阵
accuracy_rates = zeros(num_methods, length(sample_sizes)); % 预分配

set(gcf, 'unit', 'centimeters', 'position', [5 5 20 15]); % 设置整体图像大小

% 遍历每种方法
for prefix_idx = 1:num_methods
    file_prefix = file_prefixes{prefix_idx};
    
    % 遍历每个样本大小
    for i = 1:length(sample_sizes)
        file_name = sprintf('%s%d.mat', file_prefix, sample_sizes(i));
        if exist(file_name, 'file') % 确保文件存在
            data = load(file_name);
            
            % 计算正确率
            correct_count = 0;
            if isfield(data, 'result')
                for k = 1:num_runs
                    if isequal(data.result{k}, correct_matrix)
                        correct_count = correct_count + 1;
                    end
                end
            end
            accuracy_rates(prefix_idx,i) = (correct_count / num_runs) * 100; % 计算百分比
        else
            accuracy_rates(prefix_idx,i) = NaN; % 如果文件不存在，则设为空
        end
    end
    
    % 绘制子图
    subplot(2, 3, prefix_idx);
    hold on
    plot(sample_sizes, accuracy_rates(prefix_idx,:), 'o--', 'LineWidth', 1.5, 'MarkerSize', 8);
    xlabel('\itn', 'FontSize', fosize);
    ylabel('EMR (%)', 'FontSize', fosize);
    title(method_names{prefix_idx}, 'FontSize', fosize);
    ylim([0, 110]);
    set(gca, 'fontsize', fosize, 'fontname', 'Times New Roman');
    grid on;
    set(gca, 'GridLineStyle', '--');
end

set(gcf,'unit','centimeters','position',[0 0 24 12]);

lgd=legend('S1','S2');
lgd.FontName='Times New Roman';
lgd.FontSize=fosize;
lgd.Position = [0.35, 0.5, 0.1, 0.1]; % 调整图例的 [x, y, 宽度, 高度]