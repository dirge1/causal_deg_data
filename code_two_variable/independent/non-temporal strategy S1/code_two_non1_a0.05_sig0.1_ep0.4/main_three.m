clear;clc;close all
rng(1107)
t=[0:20:1000];
T=t(end);
H=0.5;
n = length(t)-1;
for wq=1:10
vec_no = wq; % no. of random vectors

for w=1:20
    cov_total = zeros(max(n)+1,max(n)+1,length(H));
    
    for f = 1:length(H)
        c = (1/(max(n)))*[0.0001,1:(length(cov_total)-1)];
        y = c.';
        cov_total(:,:,f) = bsxfun(@(c,y) 0.5.*(c.^(2*H(f))+y.^(2*H(f))-abs(y-c).^(2*H(f))), c, y);
    end
    
    %%Computation for Different n and H
    for o=1:3
        for g = 1:length(n)
            Z = normrnd(0,1,vec_no,n(g));
            
            for f = 1:length(H)
                cov = cov_total(1:n(g)+1,1:n(g)+1,f);
                
                %%Cholesky Method
                tic;
                M = chol(cov,'lower');
                B_chol{o} = M*[zeros(vec_no,1),Z]';
            end
        end
    end
    
    for i=1:vec_no
        X1_real(i,:)=normrnd(200,1)-normrnd(0.04,0.04*0.05)*t+0.1*t(end)^H.*B_chol{1}(:,i)';
        X1(i,:)=X1_real(i,:);
%         X2_real(i,:)=1/70*X1_real(i,:).^1.8-normrnd(0.05,0.05*0.05)*t+0.1*t(end)^H.*B_chol{2}(:,i)';
%         X2_real(i,:)=1/24*X1_real(i,:).^1.6-normrnd(0.05,0.05*0.05)*t+0.1*t(end)^H.*B_chol{2}(:,i)';
%         X2_real(i,:)=1/8.4*X1_real(i,:).^1.4-normrnd(0.05,0.05*0.05)*t+0.1*t(end)^H.*B_chol{2}(:,i)';
%         X2_real(i,:)=1/2.9*X1_real(i,:).^1.2-normrnd(0.05,0.05*0.05)*t+0.1*t(end)^H.*B_chol{2}(:,i)';
        X2_real(i,:)=normrnd(200,1)-normrnd(0.05,0.05*0.05)*t+0.1*t(end)^H.*B_chol{2}(:,i)';
%         X2_real(i,:)=2.9*X1_real(i,:).^0.8-normrnd(0.05,0.05*0.05)*t+0.1*t(end)^H.*B_chol{2}(:,i)';
%         X2_real(i,:)=8.4*X1_real(i,:).^0.6-normrnd(0.05,0.05*0.05)*t+0.1*t(end)^H.*B_chol{2}(:,i)';
%         X2_real(i,:)=24*X1_real(i,:).^0.4-normrnd(0.05,0.05*0.05)*t+0.1*t(end)^H.*B_chol{2}(:,i)';
%         X2_real(i,:)=70*X1_real(i,:).^0.2-normrnd(0.05,0.05*0.05)*t+0.1*t(end)^H.*B_chol{2}(:,i)';
        X2(i,:)=X2_real(i,:);
    end
    sigma_e = 0.4;
    X1 =  X1+ sigma_e*randn(size(X1)); % 5个变量的测量噪声
    X2 =  X2+ sigma_e*randn(size(X2)); % 5个变量的测量噪声
    
    X1_dif=diff(X1,[],2);
    X2_dif=diff(X2,[],2);
    
    A=X1(1,:);
    B=X2(1,:);
    TT=[A;B];
    
    fosize=16;
    if w==1
    figure
    subplot(1,2,1)
    plot(t,X1)
    xlabel('Time','fontsize',fosize,'fontname','Times New Roman')
    ylabel('X1','fontsize',fosize,'fontname','Times New Roman')
    set(gca,'fontsize',fosize,'fontname','Times New Roman');
    grid on;  % 打开网格
    set(gca, 'GridLineStyle', '--');  % 设置网格线为虚线
    
    subplot(1,2,2)
    plot(t,X2)
    xlabel('Time','fontsize',fosize,'fontname','Times New Roman')
    ylabel('X2','fontsize',fosize,'fontname','Times New Roman')
    set(gca,'fontsize',fosize,'fontname','Times New Roman');
    grid on;  % 打开网格
    set(gca, 'GridLineStyle', '--');  % 设置网格线为虚线
    
    Data{1}=X1;
    Data{2}=X2;
    end
    
    data_diff{w}=[];
    data_origin{w}=[];
    for i=1:vec_no
        data_diff{w}=[data_diff{w};X1_dif(i,:)' X2_dif(i,:)'];
        data_origin{w}=[data_origin{w};X1(i,:)' X2(i,:)'];
    end
end

% data_causal=data_causal+measurement_noise;
save(sprintf('data_numerical_var3_vec%d.mat', vec_no), 'data_diff', 'data_origin');
end