function [GCIM,pGCIM] = GCinAll(xallM,m,maketest)
% [GCIM,pGCIM] = GCinAll(xallM,m,maketest)
% GCIinAll computes the Granger Causality index (GCI) for each pair of
% time series (X,Y) taken from the input matrix 'xallM'. The VAR model for GCI
% has order 'm' and if 'maketest'=1 then parametric test is applied to
% assess the significance of GCI.
% INPUTS
% - xallM       : n x K matrix of K time series
% - m           : Order of the univariate and bivariate AR model
% - maketest    : If 1 make parametric test and give out the p-values
% OUTPUTS
% - GCIM        : K x K matrix of the Granger Causality indices
% - pGCIM       : K x K matrix of p-values of the parametric significace
%                 test for the Granger Causality index
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

if nargin==2
    maketest = 0;
end

[N,K] = size(xallM);
for iK=1:K
    xallM(:,iK) = xallM(:,iK) - mean(xallM(:,iK));
end
GCIM = NaN(K,K);
pGCIM = NaN(K,K);
for iK=1:K-1
    for jK = iK+1:K        
        % The lag matrix (at each row) for system X and Y
        xM = NaN*ones(N-m,m);
        yM = NaN*ones(N-m,m);
        for im=1:m
            xM(:,m-im+1) = xallM(im:N-1-m+im,iK);
            yM(:,m-im+1) = xallM(im:N-1-m+im,jK);
        end
        % Full regression on X from X-past and Y-past
        xbV = [xM yM]\xallM(m+1:N,iK);
        xpreV = [xM yM]*xbV;
        xerrV = xallM(m+1:N,iK) - xpreV;
        xrss1 = sum(xerrV.^2);
        % Full regression on Y from X-past and Y-past
        ybV = [xM yM]\xallM(m+1:N,jK);
        ypreV = [xM yM]*ybV;
        yerrV = xallM(m+1:N,jK) - ypreV;
        yrss1 = sum(yerrV.^2);
        % Autoregression on X from X-past
        xbV = xM\xallM(m+1:N,iK);
        xpreV = xM*xbV;
        xerrV = xallM(m+1:N,iK) - xpreV;
        xrss0 = sum(xerrV.^2);
        % Autoregression on Y from Y-past
        ybV = yM\xallM(m+1:N,jK);
        ypreV = yM*ybV;
        yerrV = xallM(m+1:N,jK) - ypreV;
        yrss0 = sum(yerrV.^2);
        
        GCIM(iK,jK) = log(yrss0/yrss1);
        GCIM(jK,iK) = log(xrss0/xrss1);
        % Compute p-values of the parametric F-test
        if maketest
            ndf = N-m-2*m;
            xfstat = ((xrss0-xrss1)/m)/(xrss1/ndf);
            pGCIM(jK,iK) = 1 - fcdf(xfstat,m,ndf);
            yfstat = ((yrss0-yrss1)/m)/(yrss1/ndf);
            pGCIM(iK,jK) = 1 - fcdf(yfstat,m,ndf);
        end
    end
end