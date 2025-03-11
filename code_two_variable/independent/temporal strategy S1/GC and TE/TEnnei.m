function [TExy,TEyx] = TEnnei(xV,yV,nnei,T,mx,taux,my,tauy)
% function [TExy,TEyx] = TEnnei(xV,yV,nnei,T,mx,taux,my,tauy)
% TEnnei estimates the transfer entropy (TE) using nearest neighbors
% (Kraskov's method). TE Indicates the influence of time series xV on time
% series yV and vice versa, given to the output.
% INPUTS
% - xV   : time series 1
% - yV   : time series 2
% - nnei : number of nearest neighbors for density estimation
% - T    : T steps ahead, note that if T>1 the whole future vector of
%          length T is considered.
% - mx   : embedding dimension for xV
% - taux : lag for xV
% - my   : embedding dimension for yV 
% - tauy : lag for yV
%OUTPUTS
% - TExy : transfer entropy from X to Y 
% - TEyx : transfer entropy from Y to X, both for the same embeddings of X and Y 

N = length(xV);
M = max((mx-1)*taux,(my-1)*tauy);
N1 = N-M-T;
% The matrix of segments (at each row) for system X
xM = NaN*ones(N1,mx);
for imx=1:mx
    xM(:,imx) = xV(M+1-(imx-1)*taux:N-T-(imx-1)*taux);
end
% The matrix of segments (at each row) for system Y
yM = NaN*ones(N1,my);
for imy=1:my
    yM(:,imy) = yV(M+1-(imy-1)*tauy:N-T-(imy-1)*tauy);
end
xpreM = NaN*ones(N1,T);
ypreM = NaN*ones(N1,T);
for iT=1:T
    xpreM(:,iT) = xV(M+1+iT:N-T+iT);
    ypreM(:,iT) = yV(M+1+iT:N-T+iT);
end
psinnei = psi(nnei); % Computed once here, to be called in several times
xM=rangescale(xM);
yM=rangescale(yM);
ypreM = rangescale(ypreM);
xpreM = rangescale(xpreM);
% TE(X->Y)
xallnowM = [ypreM xM yM];
[~, distsM] = annMaxquery(xallnowM', xallnowM', nnei+1);
maxdistV=distsM(end,:)';
n3V=nneighforgivenr(yM,maxdistV-ones(N1,1)*10^(-10));
n2V=nneighforgivenr([xM yM],maxdistV-ones(N1,1)*10^(-10));
n1V=nneighforgivenr([ypreM yM],maxdistV-ones(N1,1)*10^(-10));
psinowM = NaN*ones(N1,3);
psinowM(:,1) = psi(n1V);
psinowM(:,2) = psi(n2V);
psinowM(:,3) = -psi(n3V);
TExy = psinnei - mean(sum(psinowM,2));
% TE(Y->X|Z)
xallnowM = [xpreM yM xM];
[~, distsM] = annMaxquery(xallnowM', xallnowM', nnei+1);
maxdistV=distsM(end,:)';
n3V=nneighforgivenr(xM,maxdistV-ones(N1,1)*10^(-10));
n2V=nneighforgivenr([yM xM],maxdistV-ones(N1,1)*10^(-10));
n1V=nneighforgivenr([xpreM xM],maxdistV-ones(N1,1)*10^(-10));
psinowM = NaN*ones(N1,3);
psinowM(:,1) = psi(n1V);
psinowM(:,2) = psi(n2V);
psinowM(:,3) = -psi(n3V);
TEyx = psinnei - mean(sum(psinowM,2));

