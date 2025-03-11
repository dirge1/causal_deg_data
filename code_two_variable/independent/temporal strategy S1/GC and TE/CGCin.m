function [CGCIxy,CGCIyx,pCGCIxy,pCGCIyx] = CGCin(xV,yV,zM,m,maketest)
% [CGCIxy,CGCIyx,pCGCIxy,pCGCIyx] = CGCin(xV,yV,zM,m,maketest)
% CGCin computes the conditional Granger Causality index (GCI) for two time 
% series X and Y in the presence of other time series given as Z, for both
% directions (X->Y|Z and Y->X|Z). 
% INPUTS
% - xV          : time series 1
% - yV          : time series 2
% - zM          : the other time series accounting for 
% - m           : Order of the restricted and unrestricted AR model 
% - maketest    : If 1 make parametric test and give out the p-values
% OUTPUTS
% - CGCIxy      : Granger Causality index from X to Y in the presence of Z
% - CGCIyx      : Granger Causality index from Y to X in the presence of Z
% - pCGCIxy     : The p-value of the parametric significance test for the
%                 CGCIxy (using the F-statistic, see Econometric 
%                 Analysis, Greene, 7th Edition, Sec 5.5.2)
% - pCGCIyx     : Same as pCGCIxy but for the opposite direction
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

if nargin==4
    maketest = 0;
end

N = length(xV);
if length(yV)~=N || size(zM,1)~=N
    error('All the time series should have the same length.');
end
dz = size(zM,2);

xV = xV - mean(xV);
yV = yV - mean(yV);
zM = zM - repmat(mean(zM),N,1);

% The lag matrix (at each row) for system X and Y
xM = NaN*ones(N-m,m);
yM = NaN*ones(N-m,m);
for im=1:m
    xM(:,m-im+1) = xV(im:N-1-m+im);
    yM(:,m-im+1) = yV(im:N-1-m+im);
end
% The lag matrix (at each row) for system Z
zzM = NaN*ones(N-m,dz*m);
for id=1:dz
    colnow = (id-1)*m+1;
    for im=1:m
        zzM(:,colnow+m-im) = zM(im:N-1-m+im,id);
    end
end

% Full regression on X from X-past, Z-past and Y-past
xbV = [xM yM zzM]\xV(m+1:N);
xpreV = [xM yM zzM]*xbV;
xerrV = xV(m+1:N) - xpreV;
xrss1 = sum(xerrV.^2);
% Full regression on Y from X-past, Z-past and Y-past
ybV = [xM yM zzM]\yV(m+1:N);
ypreV = [xM yM zzM]*ybV;
yerrV = yV(m+1:N) - ypreV;
yrss1 = sum(yerrV.^2);
% Autoregression on X from X-past and Z-past
xbV = [xM zzM]\xV(m+1:N);
xpreV = [xM zzM]*xbV;
xerrV = xV(m+1:N) - xpreV;
xrss0 = sum(xerrV.^2);
% Autoregression on Y from Y-past and Z-past
ybV = [yM zzM]\yV(m+1:N);
ypreV = [yM zzM]*ybV;
yerrV = yV(m+1:N) - ypreV;
yrss0 = sum(yerrV.^2);

CGCIxy = log(yrss0/yrss1);
CGCIyx = log(xrss0/xrss1);
% Compute p-values of the parametric F-test
if maketest
    ndf = N-m-2*m-dz*m; 
    xfstat = ((xrss0-xrss1)/m)/(xrss1/ndf); 
    pCGCIyx = 1 - fcdf(xfstat,m,ndf);
    yfstat = ((yrss0-yrss1)/m)/(yrss1/ndf); 
    pCGCIxy = 1 - fcdf(yfstat,m,ndf);
else
    pCGCIxy = NaN;
    pCGCIyx = NaN;    
end
