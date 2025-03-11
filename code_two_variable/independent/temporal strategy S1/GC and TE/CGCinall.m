function [CGCIM,pCGCIM] = CGCinall(xM,m,maketest)
% [CGCIM,pCGCIM] = CGCinall(xM,m,maketest)
% CGCinall computes the conditional Granger Causality index (GCI) for all 
% time series pairs (Xi,Xj) in the presence of the rest time series in the
% vector time series given in X, for both directions (Xi->Xj|(X-{Xi,Xj} and 
% Xj->Xi|(X-{Xi,Xj}). 
% INPUTS
% - xM          : the vector time series  
% - m           : Order of the restricted and unrestricted AR model 
% - maketest    : If 1 make parametric test and give out the p-values
% OUTPUTS
% - CGCIM       : The matrix KxK of the conditional Granger Causality
%                 indexes, (i,j) for CGCI (Xi->Xj) 
% - pCGCIM      : The p-values of the parametric significance test for the
%                 values in CGCIM (using the F-statistic, see Econometric 
%                 Analysis, Greene, 7th Edition, Sec 5.5.2)
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

[N,K] = size(xM);
xM = xM - repmat(mean(xM),N,1);

% The lag matrix (at each row) for system X
xxM = NaN*ones(N-m,K*m);
for iK=1:K
    colnow = (iK-1)*m+1;
    for im=1:m
        xxM(:,colnow+m-im) = xM(im:N-1-m+im,iK);
    end
end

xrss1V = NaN*ones(K,1);
for iK=1:K
    xbV = xxM\xM(m+1:N,iK);
    xpreV = xxM*xbV;
    xerrV = xM(m+1:N,iK) - xpreV;
    xrss1V(iK) = sum(xerrV.^2);
end
CGCIM = NaN*ones(K,K);
pCGCIM = NaN*ones(K,K);
for iK=1:K
    for jK=iK+1:K
        % Autoregression on Xj from all but Xi
        irestV = setdiff([1:K*m],[(iK-1)*m+1:iK*m]);
        xbV = xxM(:,irestV)\xM(m+1:N,jK);
        xpreV = xxM(:,irestV)*xbV;
        xerrV = xM(m+1:N,jK) - xpreV;
        xrss0 = sum(xerrV.^2);
        CGCIM(iK,jK) = log(xrss0/xrss1V(jK));
        % Autoregression on Xi from all but Xj
        jrestV = setdiff([1:K*m],[(jK-1)*m+1:jK*m]);
        ybV = xxM(:,jrestV)\xM(m+1:N,iK);
        ypreV = xxM(:,jrestV)*ybV;
        yerrV = xM(m+1:N,iK) - ypreV;
        yrss0 = sum(yerrV.^2);
        CGCIM(jK,iK) = log(yrss0/xrss1V(iK));
        % Compute p-values of the parametric F-test
        if maketest
            ndf = N-m-K*m; 
            xfstat = ((xrss0-xrss1V(jK))/m)/(xrss1V(jK)/ndf); 
            pCGCIM(iK,jK) = 1 - fcdf(xfstat,m,ndf);
            yfstat = ((yrss0-xrss1V(iK))/m)/(xrss1V(iK)/ndf); 
            pCGCIM(jK,iK) = 1 - fcdf(yfstat,m,ndf);
        end
    end
end
