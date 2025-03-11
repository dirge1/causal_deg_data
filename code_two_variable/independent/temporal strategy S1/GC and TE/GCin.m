function [GCIxy,GCIyx,pGCIxy,pGCIyx] = GCin(xV,yV,m,maketest)
% [GCIxy,GCIyx,pGCIxy,pGCIyx] = GCin(xV,yV,m,maketest)
% GCin computes the Granger Causality index (GCI) for a bivariate time
% series (X,Y) for both directions (X->Y and Y->X). 
% INPUTS
% - xV          : time series 1
% - yV          : time series 2
% - m           : Order of the univariate and bivariate AR model 
% - maketest    : If 1 make parametric test and give out the p-values
% OUTPUTS
% - GCIxy : Granger Causality index from X to Y 
% - GCIyx : Granger Causality index from Y to X
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

if nargin==3
    maketest = 0;
end

N = length(xV);
if length(yV)~=N
    error('The two time series should have the same length.');
end

xV = xV - mean(xV);
yV = yV - mean(yV);

% The lag matrix (at each row) for system X and Y
xM = NaN*ones(N-m,m);
yM = NaN*ones(N-m,m);
for im=1:m
    xM(:,m-im+1) = xV(im:N-1-m+im);
    yM(:,m-im+1) = yV(im:N-1-m+im);
end
% Full regression on X from X-past and Y-past
xbV = [xM yM]\xV(m+1:N);
xpreV = [xM yM]*xbV;
xerrV = xV(m+1:N) - xpreV;
xrss1 = sum(xerrV.^2);
% Full regression on Y from X-past and Y-past
ybV = [xM yM]\yV(m+1:N);
ypreV = [xM yM]*ybV;
yerrV = yV(m+1:N) - ypreV;
yrss1 = sum(yerrV.^2);
% Autoregression on X from X-past
xbV = xM\xV(m+1:N);
xpreV = xM*xbV;
xerrV = xV(m+1:N) - xpreV;
xrss0 = sum(xerrV.^2);
% Autoregression on Y from Y-past
ybV = yM\yV(m+1:N);
ypreV = yM*ybV;
yerrV = yV(m+1:N) - ypreV;
yrss0 = sum(yerrV.^2);

GCIxy = log(yrss0/yrss1);
GCIyx = log(xrss0/xrss1);
% Compute p-values of the parametric F-test
if maketest
    ndf = N-m-2*m;
    xfstat = ((xrss0-xrss1)/m)/(xrss1/ndf); 
    pGCIyx = 1 - fcdf(xfstat,m,ndf);
    yfstat = ((yrss0-yrss1)/m)/(yrss1/ndf); 
    pGCIxy = 1 - fcdf(yfstat,m,ndf);
else
    pGCIxy = NaN;
    pGCIyx = NaN;    
end
