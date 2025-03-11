function [PTExy,PTEyx] = PTEnnei(xV,yV,zM,nnei,T,mx,taux,my,tauy,mz,tauz)
% function [PTExy,PTEyx] = PTEnnei(xV,yV,zM,nnei,T,mx,taux,my,tauy,mz,tauz)
% PTEnnei estimates the partial transfer entropy (PTE) using nearest
% neighbors (Kraskov's method). PTE indicates the direct influence of time
% series xV on time series yV accounting for the influence on yV from other
% time series contained in zM. Then the same measure is computed for the
% direct driving of xV from yV and the resulting values PTExy and PTEyx,
% respectively, are given to the output.
% INPUTS
% - xV   : time series 1, column vector
% - yV   : time series 2, column vector
% - zM   : the other time series, columnwise 
% - nnei : number of nearest neighbors for density estimation
% - T    : T steps ahead, note that if T>1 the whole future vector of
%          length T is considered.
% - mx   : embedding dimension for xV
% - taux : lag for xV
% - my   : embedding dimension for yV 
% - tauy : lag for yV
% - mz   : embedding dimension for time series in zM, the same embedding
%          dimension is used for all time series in zM. 
% - tauz : lag for time series in zM, as for mz
% OUTPUTS
% - PTExy: partial transfer entropy from X to Y 
% - PTEyx: partial transfer entropy from Y to X, both for the same
%          embeddings of X and Y 
%
%     Copyright (C) 2021 Dimitris Kugiumtzis
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%=========================================================================
% Reference : E. Siggiridou, Ch. Koutlis, A. Tsimpiris, D. Kugiumtzis, 
% "Evaluation of Granger Causality Measures for Constructing Networks from 
% Multivariate Time Series", Entropy, Vol 21 (11): 1080, 2019
% Link      : http://users.auth.gr/dkugiu/
%=========================================================================

N = size(xV,1);
if size(yV,1)~=N || size(zM,1)~=N || size(yV,2)~=1 || size(xV,2)~=1 
    error('Not proper data matrix sizes.');
end
nz = size(zM,2);
M = max([(mx-1)*taux (my-1)*tauy (mz-1)*tauz]);
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
% The matrix of segments (at each row) for system Z
zzM = NaN*ones(N1,nz*mz);
for inz=1:nz
    for imz=1:mz
        zzM(:,(inz-1)*mz+imz) = zM(M+1-(imz-1)*tauz:N-T-(imz-1)*tauz,inz);
    end
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
zzM=rangescale(zzM);
ypreM = rangescale(ypreM);
xpreM = rangescale(xpreM);
% PTE(X->Y|Z)
condM = [yM zzM];
xallnowM = [ypreM xM condM];
[~, distsM] = annMaxquery(xallnowM', xallnowM', nnei+1);
maxdistV=distsM(end,:)';
n3V=nneighforgivenr(condM,maxdistV-ones(N1,1)*10^(-10));
n2V=nneighforgivenr([xM condM],maxdistV-ones(N1,1)*10^(-10));
n1V=nneighforgivenr([ypreM condM],maxdistV-ones(N1,1)*10^(-10));
psinowM = NaN*ones(N1,3);
psinowM(:,1) = psi(n1V);
psinowM(:,2) = psi(n2V);
psinowM(:,3) = -psi(n3V);
PTExy = psinnei - mean(sum(psinowM,2));
% PTE(Y->X|Z)
condM = [xM zzM];
xallnowM = [xpreM yM condM];
[~, distsM] = annMaxquery(xallnowM', xallnowM', nnei+1);
maxdistV=distsM(end,:)';
n3V=nneighforgivenr(condM,maxdistV-ones(N1,1)*10^(-10));
n2V=nneighforgivenr([yM condM],maxdistV-ones(N1,1)*10^(-10));
n1V=nneighforgivenr([xpreM condM],maxdistV-ones(N1,1)*10^(-10));
psinowM = NaN*ones(N1,3);
psinowM(:,1) = psi(n1V);
psinowM(:,2) = psi(n2V);
psinowM(:,3) = -psi(n3V);
PTEyx = psinnei - mean(sum(psinowM,2));
