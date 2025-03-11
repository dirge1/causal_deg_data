clear;clc;close all
load data_numerical_var3_vec1_causal.mat
for wq=1:20
data=data_origin{wq};
target_variable=2;
%%
tic
P=0;
for oo=1:length(P)
    U=P(oo);

    %

    matrix_tau=1;
    matrix_E=2;

    if U>=0
        N = size(data,1);
        num_matrix = zeros(N - (matrix_E - 1) * matrix_tau, matrix_E);
        for i = 1:size(num_matrix, 1)
            for j = 1:matrix_E
                num_matrix(i, j) = i + (j - 1) * matrix_tau;
            end
        end

        for r1=1:length(matrix_tau)
            for r2=1:length(matrix_E)
                Y_estimate=[];
                Y=[];
                for p=1:size(data,2)

                    variable = data(:, p);

                    % ����Ƕ��ά����ʱ���ӳ�
                    embedding_dim =matrix_E(r2);

                    % ����ʱ�����г���

                    tau = 1;
                    reconstructed_space = zeros(N - (embedding_dim - 1) * tau, embedding_dim);
                    % ����ʱ���ӳٺ����Ƕ��ά���������ع�����
                    for i = 1:embedding_dim
                        reconstructed_space(:, i) = variable((1:N - (embedding_dim - 1) * tau) + (i - 1) * tau);
                    end

                    for q=1:size(data,2)

                        variable2 = data(:, q);

                        for o=1:size(reconstructed_space,1)-U
                            Distance = pdist2(reconstructed_space, reconstructed_space(o,:), 'euclidean')+1e-20;
                            [~,D_sort] = sort(Distance(1:end-U));
                            index_nearest = D_sort(2:embedding_dim+2);
                            u = exp(-Distance(index_nearest)/Distance(index_nearest(1)));
                            w = u/sum(u);
                            %�ٽ�������ʱ��+ʱ���ӳ�U��U>0����Ӧ������
                            Nearest_neigobor_source = data(num_matrix(index_nearest,end)+U, q);
                            weighted_neighbor_source = Nearest_neigobor_source.*w;
                            Y_estimate{p,q}(o,:) = sum(weighted_neighbor_source);
                            Y{p,q}(o,:) = data(num_matrix(o,end)+U, q);
                        end
                    end
                end
                %
                % �ڶ��е�һ��ΪY�ع�X���Դ�����
                vec=[];
                vec_estimate=[];
                R=[];
                for ii=1:size(Y,1)
                    for j=1:size(Y,2)
                        vec=Y{ii,j}(:,end);
                        vec_estimate=Y_estimate{ii,j}(:,end);
                        vec_condition=[];
                        for k=[1:j-1,j+1:size(Y,2)]
                            vec_condition=[vec_condition Y_estimate{ii,k}(:,end)];
                        end
                        %             vec_estimate=vec_estimate(:);
                        R=corr(vec_estimate,vec)';
                        R_corr{wq}(ii,j)=R;  %Ԥ���i���������������������ϵ���泤�������ı仯
                    end
                end
            end
        end
    else
        data2=data(1-U:end,:);
        N = size(data2,1);
        num_matrix = zeros(N - (matrix_E - 1) * matrix_tau, matrix_E);
        for i = 1:size(num_matrix, 1)
            for j = 1:matrix_E
                num_matrix(i, j) = i + (j - 1) * matrix_tau;
            end
        end
        num_matrix=num_matrix-U;

        for r1=1:length(matrix_tau)
            for r2=1:length(matrix_E)
                Y_estimate=[];
                Y=[];
                for p=1:size(data,2)

                    variable = data2(:, p);

                    % ����Ƕ��ά����ʱ���ӳ�
                    embedding_dim =matrix_E(r2);

                    % ����ʱ�����г���

                    tau = 1;
                    reconstructed_space = zeros(N - (embedding_dim - 1) * tau, embedding_dim);
                    % ����ʱ���ӳٺ����Ƕ��ά���������ع�����
                    for i = 1:embedding_dim
                        reconstructed_space(:, i) = variable((1:N - (embedding_dim - 1) * tau) + (i - 1) * tau);
                    end

                    for q=1:size(data,2)

                        variable2 = data2(:, q);

                        for o=1:size(reconstructed_space,1)
                            Distance = pdist2(reconstructed_space, reconstructed_space(o,:), 'euclidean')+1e-20;
                            [~,D_sort] = sort(Distance(1:end));
                            index_nearest = D_sort(2:embedding_dim+2);
                            u = exp(-Distance(index_nearest)/Distance(index_nearest(1)));
                            w = u/sum(u);
                            %�ٽ�������ʱ��+ʱ���ӳ�U��U<0����Ӧ������
                            Nearest_neigobor_source = data(num_matrix(index_nearest,end)+U, q);
                            weighted_neighbor_source = Nearest_neigobor_source.*w;
                            Y_estimate{p,q}(o,:) = sum(weighted_neighbor_source);
                            Y{p,q}(o,:) = data(num_matrix(o,end)+U, q);
                        end
                    end
                end
                %
                % �ڶ��е�һ��ΪY�ع�X���Դ�����
                vec=[];
                vec_estimate=[];
                R=[];
                for ii=1:size(Y,1)
                    for j=1:size(Y,2)
                        vec=Y{ii,j}(:,end);
                        vec_estimate=Y_estimate{ii,j}(:,end);
                        vec_condition=[];
                        for k=[1:j-1,j+1:size(Y,2)]
                            vec_condition=[vec_condition Y_estimate{ii,k}(:,end)];
                        end

                        R=corr(vec_estimate,vec)';
                        R_corr{wq}(ii,j)=R;  %Ԥ���i���������������������ϵ���泤�������ı仯

                        
                    end
                end
            end
        end
    end
end
%
end

for i=1:20
    CCM_1_2(i)=R_corr{i}(2,1);
    CCM_2_1(i)=R_corr{i}(1,2);
end

fosize=16;

figure
plot(CCM_1_2,'o--')
hold on
plot(CCM_2_1,'s--')
ylabel('Causal strength', 'fontsize', fosize, 'fontname', 'Times New Roman');
xlabel('Test number', 'fontsize', fosize, 'fontname', 'Times New Roman');
grid on;  % ������
    set(gca, 'GridLineStyle', '--');  % ����������Ϊ����
% ��������
set(gca, 'fontsize', fosize, 'fontname', 'Times New Roman');
set(gcf,'unit','centimeters','position',[10 5 12 9]);

lgd=legend('X1 to X2','X2 to X1');
lgd.Location='northwest';
lgd.FontName='Times New Roman';
lgd.FontSize=fosize;
ylim([0.9945,1])
%% 
% clear;clc;close all
% 
% M = 1;
% 
% xx(1:M,1)=0.2;
% yy(1:M,1)=0.4;
% rx =3.78;
% ry = 3.77;
% beitaxy=0.01;
% beitayx=0;
% LL=4000;
% % LL=5;
% for oo=1:length(LL)
%     for i=M:LL(oo)
%         xx(i+1,1)=xx(i)*(rx-rx*xx(i)-beitaxy*yy(i));
%         yy(i+1,1)=yy(i)*(ry-ry*yy(i)-beitayx*xx(i-M+1));
%     end
%     
%     x=xx+normrnd(0,0,[length(xx),1]);
%     y=yy+normrnd(0,0,[length(yy),1]);
% 
%     data=[x y];
% % target_variable=2;
% %%
% tic
% P=0;
% for oo=1:length(P)
%     U=P(oo);
% 
%     %
% 
%     matrix_tau=1;
%     matrix_E=2;
% 
%     if U>=0
%         N = size(data,1);
%         num_matrix = zeros(N - (matrix_E - 1) * matrix_tau, matrix_E);
%         for i = 1:size(num_matrix, 1)
%             for j = 1:matrix_E
%                 num_matrix(i, j) = i + (j - 1) * matrix_tau;
%             end
%         end
% 
%         for r1=1:length(matrix_tau)
%             for r2=1:length(matrix_E)
%                 Y_estimate=[];
%                 Y=[];
%                 for p=1:size(data,2)
% 
%                     variable = data(:, p);
% 
%                     % ����Ƕ��ά����ʱ���ӳ�
%                     embedding_dim =matrix_E(r2);
% 
%                     % ����ʱ�����г���
% 
%                     tau = 1;
%                     reconstructed_space = zeros(N - (embedding_dim - 1) * tau, embedding_dim);
%                     % ����ʱ���ӳٺ����Ƕ��ά���������ع�����
%                     for i = 1:embedding_dim
%                         reconstructed_space(:, i) = variable((1:N - (embedding_dim - 1) * tau) + (i - 1) * tau);
%                     end
% 
%                     for q=1:size(data,2)
% 
%                         variable2 = data(:, q);
% 
%                         for o=1:size(reconstructed_space,1)-U
%                             Distance = pdist2(reconstructed_space, reconstructed_space(o,:), 'euclidean')+1e-20;
%                             [~,D_sort] = sort(Distance(1:end-U));
%                             index_nearest = D_sort(2:embedding_dim+2);
%                             u = exp(-Distance(index_nearest)/Distance(index_nearest(1)));
%                             w = u/sum(u);
%                             %�ٽ�������ʱ��+ʱ���ӳ�U��U>0����Ӧ������
%                             Nearest_neigobor_source = data(num_matrix(index_nearest,end)+U, q);
%                             weighted_neighbor_source = Nearest_neigobor_source.*w;
%                             Y_estimate{p,q}(o,:) = sum(weighted_neighbor_source);
%                             Y{p,q}(o,:) = data(num_matrix(o,end)+U, q);
%                         end
%                     end
%                 end
%                 %
%                 % �ڶ��е�һ��ΪY�ع�X���Դ�����
%                 vec=[];
%                 vec_estimate=[];
%                 R=[];
%                 for ii=1:size(Y,1)
%                     for j=1:size(Y,2)
%                         vec=Y{ii,j}(:,end);
%                         vec_estimate=Y_estimate{ii,j}(:,end);
%                         vec_condition=[];
%                         for k=[1:j-1,j+1:size(Y,2)]
%                             vec_condition=[vec_condition Y_estimate{ii,k}(:,end)];
%                         end
%                         %             vec_estimate=vec_estimate(:);
%                         R=corr(vec_estimate,vec)';
%                         R_corr(ii,j)=R;  %Ԥ���i���������������������ϵ���泤�������ı仯
%                     end
%                 end
%             end
%         end
%     else
%         data2=data(1-U:end,:);
%         N = size(data2,1);
%         num_matrix = zeros(N - (matrix_E - 1) * matrix_tau, matrix_E);
%         for i = 1:size(num_matrix, 1)
%             for j = 1:matrix_E
%                 num_matrix(i, j) = i + (j - 1) * matrix_tau;
%             end
%         end
%         num_matrix=num_matrix-U;
% 
%         for r1=1:length(matrix_tau)
%             for r2=1:length(matrix_E)
%                 Y_estimate=[];
%                 Y=[];
%                 for p=1:size(data,2)
% 
%                     variable = data2(:, p);
% 
%                     % ����Ƕ��ά����ʱ���ӳ�
%                     embedding_dim =matrix_E(r2);
% 
%                     % ����ʱ�����г���
% 
%                     tau = 1;
%                     reconstructed_space = zeros(N - (embedding_dim - 1) * tau, embedding_dim);
%                     % ����ʱ���ӳٺ����Ƕ��ά���������ع�����
%                     for i = 1:embedding_dim
%                         reconstructed_space(:, i) = variable((1:N - (embedding_dim - 1) * tau) + (i - 1) * tau);
%                     end
% 
%                     for q=1:size(data,2)
% 
%                         variable2 = data2(:, q);
% 
%                         for o=1:size(reconstructed_space,1)
%                             Distance = pdist2(reconstructed_space, reconstructed_space(o,:), 'euclidean')+1e-20;
%                             [~,D_sort] = sort(Distance(1:end));
%                             index_nearest = D_sort(2:embedding_dim+2);
%                             u = exp(-Distance(index_nearest)/Distance(index_nearest(1)));
%                             w = u/sum(u);
%                             %�ٽ�������ʱ��+ʱ���ӳ�U��U<0����Ӧ������
%                             Nearest_neigobor_source = data(num_matrix(index_nearest,end)+U, q);
%                             weighted_neighbor_source = Nearest_neigobor_source.*w;
%                             Y_estimate{p,q}(o,:) = sum(weighted_neighbor_source);
%                             Y{p,q}(o,:) = data(num_matrix(o,end)+U, q);
%                         end
%                     end
%                 end
%                 %
%                 % �ڶ��е�һ��ΪY�ع�X���Դ�����
%                 vec=[];
%                 vec_estimate=[];
%                 R=[];
%                 for ii=1:size(Y,1)
%                     for j=1:size(Y,2)
%                         vec=Y{ii,j}(:,end);
%                         vec_estimate=Y_estimate{ii,j}(:,end);
%                         vec_condition=[];
%                         for k=[1:j-1,j+1:size(Y,2)]
%                             vec_condition=[vec_condition Y_estimate{ii,k}(:,end)];
%                         end
% 
%                         R=corr(vec_estimate,vec)';
%                         R_corr(ii,j)=R;  %Ԥ���i���������������������ϵ���泤�������ı仯
% 
%                         
%                     end
%                 end
%             end
%         end
%     end
% end
% end
% 
