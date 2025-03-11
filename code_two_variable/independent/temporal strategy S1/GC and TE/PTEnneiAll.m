function PTEM = PTEnneiAll(xallM,nnei,T,m,tau)
% function PTEM = PTEnneiAll(xallM,nnei,T,m,tau)
% PTEnneiAll estimates the partial transfer entropy (PTE) using nearest
% neighbors (Kraskov's method) for all pair of variables. The time series
% are in the input matrix 'xallM'.
% PTE indicates the direct influence of time series xV on time series yV
% accounting for the influence on yV from the other time series in 'xallM'.
% INPUTS
% - xallM: N x K matrix of the K time series
% - nnei : number of nearest neighbors for density estimation
% - T    : T steps ahead, note that if T>1 the whole future vector of
%          length T is considered.
% - m    : embedding dimension
% - tau  : lag
% OUTPUTS
% - PTEM : K x K matrix of the values of partial transfer entropy

[N,K] = size(xallM);
M = (m-1)*tau;
N1 = N-M-T;
PTEM = NaN(K,K);
for iK=1:K
    % The matrix of segments (at each row) for system X
    xM = NaN*ones(N1,m);
    for im=1:m
        xM(:,im) = xallM(M+1-(im-1)*tau:N-T-(im-1)*tau,iK);
    end
    xpreM = NaN*ones(N1,T);
    for iT=1:T
        xpreM(:,iT) = xallM(M+1+iT:N-T+iT,iK);
    end
    for jK=1:K
        % The matrix of segments (at each row) for system Y
        yM = NaN*ones(N1,m);
        for im=1:m
            yM(:,im) = xallM(M+1-(im-1)*tau:N-T-(im-1)*tau,jK);
        end
        ypreM = NaN*ones(N1,T);
        for iT=1:T
            ypreM(:,iT) = xallM(M+1+iT:N-T+iT,jK);
        end
        % The matrix of segments (at each row) for system Z
        itmpV = setdiff([1:K],[iK jK]);
        L = K-2;
        zM = xallM(:,itmpV);
        zzM = NaN*ones(N1,L*m);
        for iL=1:L
            for im=1:m
                zzM(:,(iL-1)*m+im) = zM(M+1-(im-1)*tau:N-T-(im-1)*tau,iL);
            end
        end
        ypreM = NaN*ones(N1,T);
        for iT=1:T
            ypreM(:,iT) = xallM(M+1+iT:N-T+iT,jK);
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
        PTEM(iK,jK) = psinnei - mean(sum(psinowM,2));
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
        PTEM(jK,iK) = psinnei - mean(sum(psinowM,2));
    end
end
