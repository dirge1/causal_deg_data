function npV=nneighforgivenr(xM,rV)
% npV=nneighforgivenr(xM,rV)
% The function nneighforgivenr uses the k-d-tree data structure to find the
% number of nearest neighbors of each target point that are within a
% given distance. The target points are given in the rows of 'xM' and the
% distances in the vector 'rV'.  
% INPUTS
% - xM          : matrix n x K, the vector time series  
% - rV          : vector n x 1, the distances for each target point in xM
% OUTPUTS
% - npV         : vector n x 1, the number of nearest neighbors for each
%                 target point in xM within the given distance in rV 
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

npV = annMaxRvaryquery(xM', xM',rV, 1, 'search_sch', 'fr', 'radius', sqrt(1));
npV=double(npV);
npV(npV==0)=1;
