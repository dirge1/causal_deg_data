https://github.com/dkugiu/Matlab
Created on 18/1/2021 by Dimitris Kugiumtzis

The files in this directory are for various measures of Granger causality: 
1. Granger causality index (GCI)
2. conditional Granger causality index (CGCI)
3. transfer entropy (TE) [using the nearest neighbor estimate]
4. partial transfer entropy (PTE) [using the nearest neighbor estimate]

For each measures there are two Matlab files that can be called, a) one for the causality of a specific pair of variables in both directions, Xi -> Xj and Xj -> Xi, and b) another for the causality of all pairs of variables of the multivariate time series of K variables. The TE and PTE call mex-files compiled for Windows and linux (no Mac executables) for the nearest neighbor search. The user may want to replace the routines for nearest neighbor search in the Matlab files with the Matlab (home-made) routines. We found that in general Matlab routines are slower than the ones implemented here. The routines for k-d-tree are from a distribution of the so-called annquery (found once in the internet, but could not trace it anymore).  

List of files in the directory:
1. a) GCin.m, b) GCinAll.m
2. a) CGCin.m, b) CGCinAll.m
3. a) TEnnei.m, b) TEnneiAll.m
4. a) PTEnnei.m, b) PTEnneiAll.m
   
Auxiliary files:
- nneighforgivenr.m  called by TE and PTE functions to search for nearest neighbors within a given distance.
- annMaxRvaryquery.m  the main function for searching for nearest neighbors using the k-d-tree structure. The function calls the executable mex-files in the directory, compiled for win-32, win-64 and linux-64. 
- setopts.m  goes along with the functions for nearest neighbor search
- rangescale.m  scaling the data to [0,1] before computing TE and PTE.

Please remember to cite when appropriate:
%=========================================================================
% Reference : E. Siggiridou, Ch. Koutlis, A. Tsimpiris, D. Kugiumtzis, 
% "Evaluation of Granger Causality Measures for Constructing Networks from 
% Multivariate Time Series", Entropy, Vol 21 (11): 1080, 2019
% Link      : http://users.auth.gr/dkugiu/
%=========================================================================

