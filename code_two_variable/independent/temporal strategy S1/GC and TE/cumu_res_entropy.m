function F=cumu_res_entropy(a)
A=sort(a);
cd=length(A);
n=(A(2:end)-A(1:end-1)).*((1-(1:cd-1)./cd).*log(1-(1:cd-1)./cd));
F=-sum(n);
end