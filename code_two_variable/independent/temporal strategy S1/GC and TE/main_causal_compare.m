clear;clc;close all
load('data_numerical_var3_vec1.mat')
for wq=1:20
    data=data_origin{wq};
    for P=1:1
        %     P
        data0=[];
        data0(:,1)=data(1:end-P+1,1);
        data0(:,2)=data(1+P-1:end,2);
        [GCIM{wq},pGCIM] = GCinAll(data0,3);
        %     [CGCIM{P},pCGCIM] = CGCinall(data0,3);
        TEM{wq} = TEnneiAll(data0,3,1,2,1);
        %     PTEM{P} = PTEnneiAll(data0,3,1,2,1);
    end
end

for i=1:20
    GC_1_2(i)=GCIM{i}(1,2);
    GC_2_1(i)=GCIM{i}(2,1);
end

for i=1:20
    TE_1_2(i)=TEM{i}(1,2);
    TE_2_1(i)=TEM{i}(2,1);
end

fosize=16;

%% 
figure
plot(GC_1_2,'o--')
hold on
plot(GC_2_1,'s--')
ylabel('Causal strength', 'fontsize', fosize, 'fontname', 'Times New Roman');
xlabel('Test number', 'fontsize', fosize, 'fontname', 'Times New Roman');
grid on;  % 打开网格
    set(gca, 'GridLineStyle', '--');  % 设置网格线为虚线
% 设置字体
set(gca, 'fontsize', fosize, 'fontname', 'Times New Roman');
set(gcf,'unit','centimeters','position',[10 5 12 9]);

lgd=legend('X1 to X2','X2 to X1');
lgd.Location='northwest';
lgd.FontName='Times New Roman';
lgd.FontSize=fosize;
ylim([0,1])

figure
plot(TE_1_2,'o--')
hold on
plot(TE_2_1,'s--')
ylabel('Causal strength', 'fontsize', fosize, 'fontname', 'Times New Roman');
xlabel('Test number', 'fontsize', fosize, 'fontname', 'Times New Roman');
grid on;  % 打开网格
    set(gca, 'GridLineStyle', '--');  % 设置网格线为虚线
% 设置字体
set(gca, 'fontsize', fosize, 'fontname', 'Times New Roman');
set(gcf,'unit','centimeters','position',[10 5 12 9]);

lgd=legend('X1 to X2','X2 to X1');
lgd.Location='northwest';
lgd.FontName='Times New Roman';
lgd.FontSize=fosize;
ylim([0,0.15])
%% 
% M = 1;
%
% xx(1:M,1)=0.2;
% yy(1:M,1)=0.4;
% rx =3.78;
% ry = 3.77;
% beitaxy=0;
% beitayx=0.5;
% % LL=1000:500:5000;
% LL=5000;
% for o=1:length(LL)
%     for i=M:LL(o)
%         xx(i+1,1)=xx(i)*(rx-rx*xx(i)-beitaxy*yy(i));
%         yy(i+1,1)=yy(i)*(ry-ry*yy(i)-beitayx*xx(i-M+1));
%     end
%
%     x = xx + normrnd(0, 0, [length(xx), 1]);
%     y = yy + normrnd(0, 0, [length(yy), 1]);
%
%     data0=[x y];
%     [GCIM{o}] = GCinAll(data0,3);
% %     [CGCIM{P},pCGCIM] = CGCinall(data0,3);
%     TEM{o} = TEnneiAll(data0,3,1,2,1);
% end


