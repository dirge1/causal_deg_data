function TEM = TEnneiAll(xallM,nnei,T,m,tau)
% function TEM = TEnneiAll(xallM,nnei,T,m,tau)
% TEnneiAll estimates the transfer entropy (TE) using nearest neighbors
% (Kraskov's method) for all pair of time series from the given matrix
% 'xallM'.  
% TE Indicates the influence of time series xV on time
% series yV and vice versa, given to the output.
% INPUTS
% - xallM: n x K matrix of K time series 
% - nnei : number of nearest neighbors for density estimation
% - T    : T steps ahead, note that if T>1 the whole future vector of
%          length T is considered.
% - m    : embedding dimension 
% - tau  : lag 
%OUTPUTS
% - TEM  : K x K matrix of the values of transfer entropy

[N,K] = size(xallM);
M = (m-1)*tau;
N1 = N-M-T;
TEM = NaN(K,K);
for iK=1:K-1
    % The matrix of segments (at each row) for system X
    xM = NaN*ones(N1,m);
    xpreM = NaN*ones(N1,T);
    for im=1:m
        xM(:,im) = xallM(M+1-(im-1)*tau:N-T-(im-1)*tau,iK);
    end
    for iT=1:T
        xpreM(:,iT) = xallM(M+1+iT:N-T+iT,iK);
    end
    for jK=iK+1:K
        % The matrix of segments (at each row) for system Y
        yM = NaN*ones(N1,m);
        ypreM = NaN*ones(N1,T);
        for im=1:m
            yM(:,im) = xallM(M+1-(im-1)*tau:N-T-(im-1)*tau,jK);
        end
        for iT=1:T
            ypreM(:,iT) = xallM(M+1+iT:N-T+iT,jK);
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
        TEM(iK,jK) = psinnei - mean(sum(psinowM,2));
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
        TEM(jK,iK) = psinnei - mean(sum(psinowM,2));
    end
end
