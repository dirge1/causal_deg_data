clear;clc;close all
% rng(37)
t=[0:20:1000];
T=t(end);
H=0.5;
n = length(t)-1;
vec_no = 200; % no. of random vectors

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
    X1_real(i,:)=normrnd(200,1)-normrnd(0.03,0.03*0.05)*t.^0.8+0.02*t(end)^H.*B_chol{1}(:,i)';
    X1(i,:)=X1_real(i,:);
    X2_real(i,:)=normrnd(200,1)-normrnd(0.05,0.05*0.05)*t.^0.7+0.02*t(end)^H.*B_chol{2}(:,i)';
    X2(i,:)=X2_real(i,:);
    X3_real(i,:)=(X1_real(i,:).^0.5+X2_real(i,:).^0.5)*10-normrnd(0.03,0.03*0.05)*t.^0.5+0.02*t(end)^H.*B_chol{1}(:,i)';
    X3(i,:)=X3_real(i,:);
end

A=X1(1,:);
B=X2(1,:);
C=X3(1,:);
TT=[A;B;C];

fosize=16;

figure
subplot(1,3,1)
plot(t,X1)
xlabel('Time','fontsize',fosize,'fontname','Times New Roman')
ylabel('Y1','fontsize',fosize,'fontname','Times New Roman')
set(gca,'fontsize',fosize,'fontname','Times New Roman');
grid on;  % 打开网格
set(gca, 'GridLineStyle', '--');  % 设置网格线为虚线

subplot(1,3,2)
plot(t,X2)
xlabel('Time','fontsize',fosize,'fontname','Times New Roman')
ylabel('Y2','fontsize',fosize,'fontname','Times New Roman')
set(gca,'fontsize',fosize,'fontname','Times New Roman');
grid on;  % 打开网格
set(gca, 'GridLineStyle', '--');  % 设置网格线为虚线

subplot(1,3,3)
plot(t,X3)
Data{1}=X1;
Data{2}=X2;
Data{3}=X3;

xlabel('Time(h)','fontsize',fosize,'fontname','Times New Roman')
ylabel('Y3','fontsize',fosize,'fontname','Times New Roman')
set(gca,'fontsize',fosize,'fontname','Times New Roman');
set(gcf,'unit','centimeters','position',[0 0 24 12]);
grid on;  % 打开网格
set(gca, 'GridLineStyle', '--');  % 设置网格线为虚线

% X3=X3+0.2*randn(vec_no, 1);

data_causal=[];
for i=10
    data_causal=[X1(:,i) X2(:,i) X3(:,i)];
end
measurement_noise =  0.2*randn(vec_no, 3); % 5个变量的测量噪声
data_causal=data_causal+measurement_noise;
save('data_numerical','data_causal')