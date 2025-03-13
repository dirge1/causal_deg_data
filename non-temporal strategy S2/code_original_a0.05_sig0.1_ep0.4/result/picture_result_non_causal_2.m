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

sample_sizes = 1:10; % ������
num_methods = length(file_prefixes);
num_runs = 20; % ÿ�����������д���
fosize = 14; % �����С
correct_matrix = [0 0; 0 0]; % ��������ȷ����
accuracy_rates = zeros(num_methods, length(sample_sizes)); % Ԥ����

set(gcf, 'unit', 'centimeters', 'position', [5 5 20 15]); % ��������ͼ���С

% ����ÿ�ַ���
for prefix_idx = 1:num_methods
    file_prefix = file_prefixes{prefix_idx};
    
    % ����ÿ��������С
    for i = 1:length(sample_sizes)
        file_name = sprintf('%s%d.mat', file_prefix, sample_sizes(i));
        if exist(file_name, 'file') % ȷ���ļ�����
            data = load(file_name);
            
            % ������ȷ��
            correct_count = 0;
            if isfield(data, 'result')
                for k = 1:num_runs
                    if isequal(data.result{k}, correct_matrix)
                        correct_count = correct_count + 1;
                    end
                end
            end
            accuracy_rates(prefix_idx,i) = (correct_count / num_runs) * 100; % ����ٷֱ�
        else
            accuracy_rates(prefix_idx,i) = NaN; % ����ļ������ڣ�����Ϊ��
        end
    end
    
    % ������ͼ
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
lgd.Position = [0.35, 0.5, 0.1, 0.1]; % ����ͼ���� [x, y, ���, �߶�]