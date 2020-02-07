function [Omega,S,loglik,iter] = identKron8(X,p,nl,ng,e,tol,verb,flag_init)
% Optimization sequence: Sigma, Lambda, Gamma
%identGMrw computes the sparse grafical model using the re-weigthed scheme
%
% Inputs:  X data (row is the time, columns are the components)
%          p order of the AR model
%          e epsilon parameter (>0) in the re-weigthed scheme
%          tol tolerance for stopping criterium of the re-weighted
%          iterations
%          verb 'v'=verbouse modality, ' '=no verbose modality
%
% Outputs: Omega sparsity patters (0 = no edge, 1 = edge)
%          S=[S0 S1 ... Sn] coefficients of the partial coherence function
%
%          S = S0+0.5sum_{k=1}^n S_k z^{-k}+S_k' z^{k}  
%          loglik negative log-likelihood at the optimum 
%          iter number of iterations for the reweighted
% 
% In this case the reweighting is with GML  
%

% Code written by Mattia Zorzi 2019
% Please cite the corresponding paper [Mattia Zorzi. "Autoregressive 
% Identification of Kronecker Graphical Models", 2020]


%% variables
N = size(X,1);
n = size(X,2);

%% weight matrix
T=zeros(n,n);
for h=1:nl
    for j=1:nl
        for kk=1:ng
            for l=1:ng
                if h==j & kk==l
                    T((h-1)*ng+kk,(j-1)*ng+l) =p+1;
                end
                if h==j & kk<l
                    T((h-1)*ng+kk,(j-1)*ng+l) =2*p+1;
                end
                if h<j & kk==l
                    T((h-1)*ng+kk,(j-1)*ng+l) =2*p+1;
                end
                if h<j & kk<l
                    T((h-1)*ng+kk,(j-1)*ng+l) =4*p+2;
                end
 
            end
        end
    end
end



%% init gammas
T_C=symm(sTop(X,p));
L=chol(T_C);
Li=L^-1;
B=Li*Li'*[eye(n); zeros(n*p,n)];
B=B*sqrtm(B(1:n,1:n))^-1;
Sinit = get_D_short(B*B', n, p);
L1=ones(nl,nl);
G1=ones(ng,ng);
if nargin==8 & flag_init=='init'
    dd=10;
    while dd>1e-4
        L1o=L1;
        G1o=G1;
        [L1,G1] = reweight(Sinit,L1,G1,T,e,nl,ng);
        dd = norm(L1-L1o)+norm(G1-G1o) ;
    end
end
Lambda(:,:,1)=L1;
Gamma(:,:,1)=G1;

   
%% thresholding in the final inverse
thr = 10^-4; 

%% optimize
k=1;
d=1;
while d>=tol
    W = symm(2/(N-p)*max(kron(Lambda(:,:,k),ones(ng,ng)),kron(ones(nl,nl),Gamma(:,:,k))));  
    [Omega,Sp{k},fit,reg] = sparseGMW2(X,W,n,nl,ng,p,thr,verb);
    [Lambda(:,:,k+1), Gamma(:,:,k+1)]=reweight(Sp{k},Lambda(:,:,k),Gamma(:,:,k),T,e,nl,ng);
    if k>1
        d = distance(Sp{k-1},Sp{k});
    end
    loglik = (N-p)/2*(fit+ reg)- sum(sum(T.*log(W*(N-p)/2)))+e*sum(sum(triu(Lambda(:,:,k))))+e*sum(sum(triu(Gamma(:,:,k))));
    k=k+1;
end    

iter = k-1;  % totoal iterations
Lambda = Lambda(:,:,1:end-1);
Gamma = Gamma(:,:,1:end-1);
S=Sp{end};

end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   FUNCTIONS USED IN MAIN CODE   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Omega,S,fit,reg,Z] = sparseGMW2(X,gamma,n,nl,ng,p,thr,verb)
%SPARSEGMW computes the sparse grafical model
%
% Inputs:  X data (row is the time, columns are the components)
%          gamma (positive entrywise and symmetric) regularization
%          parameter maitrx
%          n dimension of the process
%          p order of the AR model
%          thr (<1) thresholding in the partial coherence
%          verb 'v'=verbouse modality, ' '=no verbose modality
%
% Outputs: Omega sparsity patters (0 = no edge, 1 = edge)
%          A=[A0 A1 ... Ap] matrix coefficients of the AR model
%        
%             sum_{k=0}^n A_k y(t-k) =e(t),   
%
%          e(t) WGN with covaraince matrix I_n
%          R partial coherence function (th is the vector of frequencies)
%          S=[S0 S1 ... Sn] coefficients of the partial coherence function
%
%          S = S0+0.5sum_{k=1}^n S_k z^{-k}+S_k' z^{k}   
% 


%% parameter in the armijo condition
alpha = 0.1;

%% stopping tolerance for the duality gap
tol=10^-4;

%% maximum number of iterations
max_iter=3000;

%% sample Toeplitz covariance
C=symm(sTop(X,p));

%% init
Z = zeros(n,n,p+1);
obj = phi(C,Z);
t = 0.2;
dgap = tol+1;
it=0;

%% optimization with repsect to Z
while dgap >= tol
    it=it+1;
    if it>max_iter
        break
    end 
    gradZ = grad_phi(C,Z);
    t = min(2.5*t,0.5);
    flagp = false;
    while not(flagp)
        t = t/2;
        Zn = proj_C1Kron1(Z-t*gradZ,gamma,nl,ng);
        flagp = check_positive_definiteness(C,Zn);
        if t<10^-6
            break
        end
    end
    t = 2*t;
    flaga = false;
    while not(flaga)
        t = t/2;
        Zn = proj_C1Kron1(Z-t*gradZ,gamma,nl,ng);
        flaga = check_armijo(obj,C,Z,Zn,gradZ,alpha,t);
        if t<10^-6
            break
        end        
    end
    Z = Zn;
    obj = phi(C,Z);
    dgap = dual_gap(C, Z, gamma,nl,ng);
    if verb=='v'
        disp(['iteration#' num2str(it) '    obj dual:' num2str(obj) '   duality gap: ' num2str(dgap) ])
    end
end
    
%% primal solution
[dgap,X] = dual_gap(C, Z, gamma,nl,ng);

% coeffieicients of the inverse of the PSD
S = get_D_short(X, n, p);
S(:,:,1)=symm(S(:,:,1));


% inverse of the PSD 
th = linspace(0,pi,200);
R = zeros(n,n,200);
for k=1:size(th,2)
    SS=S(:,:,1);
    for j=2:size(S,3)
        SS = SS+0.5*(S(:,:,j)*exp(-sqrt(-1)*th(k)*(j-1))+S(:,:,j)'*exp(sqrt(-1)*th(k)*(j-1))); 
    end
    R(:,:,k) = SS;
end

%% sparsity patter with thresholding
Omega = zeros(n,n);
for k=1:n
    for j=1:n
        Omega(j,k) = max(abs(R(j,k,:)));
    end
end
Omega = symm(Omega);
Omega = Omega>thr;


%% computes the fit and regularization term

[fit , reg] = fit_vs_reg(C, Z, gamma,nl,ng);



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function r = phi(C, Z)
% Function to evaluate phi(C+T(Z)), defined in the paper
    T_Z = get_T(Z);
    n = size(Z,1);
    V = C + T_Z;
    V00 = V(1:n,1:n);
    V1p0 = V(n+1:end,1:n);
    V1p1p = V(n+1:end,n+1:end);
    V_temp = V00 - V1p0'*(V1p1p\V1p0);
    V_temp = symm(V_temp);
    r = -log(det(V_temp));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function grad_Z = grad_phi(C, Z)
% Function to compute the gradient of phi
    n = size(Z, 1);
    p = size(Z, 3) - 1;
    T_Z=get_T(Z);
    V = C + T_Z;
   
%     L=chol(symm(V),'lower');
%     L00 = L(1:n,1:n);
%     L1p0 = L(n+1:end,1:n);
%     L1p1p = L(n+1:end,n+1:end);
%     Ltmp=[eye(n); -L1p1p^-1*L1p0]*L00^-1;
%     grad_Z =- get_D_short(Ltmp*Ltmp', n, p);
    
    Y = sparse([zeros(n*p,n) eye(n*p)]);
    T1 = speye(size(V))/V;
    T2Y = ((Y*V)*Y')\Y;     
    grad_Z = get_D_short(-T1 + Y'*T2Y, n, p);

 
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [h_BM] = get_hW(D,gamma,nl,ng)
% Function to compute the weighted "infinity norm".
% D is in a 3d format.
    Dabs = abs(D);
    Dabs_max_matrix = gamma.*max(Dabs, [], 3);
    W=zeros(nl*ng,nl*ng);
    for k=1:nl
        for j=k:nl
            W1 = Dabs_max_matrix((k-1)*ng+1:k*ng,(j-1)*ng+1:j*ng);
            W2 = Dabs_max_matrix((j-1)*ng+1:j*ng,(k-1)*ng+1:k*ng);
            W1 = triu(max(W1,W1'));
            W2 = triu(max(W2,W2'));
            W((k-1)*ng+1:k*ng,(j-1)*ng+1:j*ng)=max(W1,W2);
        end
    end
    h_BM = sum(sum(W));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Z_projected = proj_C1Kron1(Z, gamma,nl,ng) %Z is the U of the equations p.2699
% Function to project onto the C1 constraint

    p = size(Z, 3) - 1;
    X = zeros(1, 4*(p+1)); % A row vector
    Z_projected = zeros(size(Z));
    Z(:,:,1) = symm(Z(:,:,1)); % Ensure that the first block is symmetric
    
    % Reformulation using vectors
    for j = 1 : nl
        for k = j : nl % For each element we perform LASSO
            for i = 1 : ng
                for l = i : ng 
                    if j == k
                        if i == l
                            gamma((j-1)*ng+i,(k-1)*ng+l) = 4*gamma((j-1)*ng+i,(k-1)*ng+l);
                        else 
                            gamma((j-1)*ng+i,(k-1)*ng+l) = 2*gamma((j-1)*ng+i,(k-1)*ng+l);
                        end
                    else
                        if i == l
                            gamma((j-1)*ng+i,(k-1)*ng+l) = 2*gamma((j-1)*ng+i,(k-1)*ng+l);
                        end
                    end
                    X(1 : p+1) = Z((j-1)*ng+i,(k-1)*ng+l,:);
                    X(p+2 : 2*(p+1)) = Z((j-1)*ng+l,(k-1)*ng+i,:);
                    X(2*(p+1)+1 : 3*(p+1)) = Z((k-1)*ng+i,(j-1)*ng+l,:);
                    X(3*(p+1)+1 : 4*(p+1)) = Z((k-1)*ng+l,(j-1)*ng+i,:);
                    absX = abs(X);
                    Z_temp = projectSortC(absX', gamma((j-1)*ng+i,(k-1)*ng+l)); % Using the mex file from EWOUT VAN DEN BERG, MARY SCHMIDT, MICHAEL P. FRIEDLANDER, AND YEVIN MURPHY
                    Z_temp = (sign(X').*Z_temp)';
                    Z_projected((j-1)*ng+i,(k-1)*ng+l,:) = Z_temp(1 : p+1);
                    Z_projected((j-1)*ng+l,(k-1)*ng+i,:) = Z_temp(p+2 : 2*(p+1));
                    Z_projected((k-1)*ng+i,(j-1)*ng+l,:) = Z_temp(2*(p+1)+1 : 3*(p+1));
                    Z_projected((k-1)*ng+l,(j-1)*ng+i,:) = Z_temp(3*(p+1)+1 : 4*(p+1));
                end
            end           
            
        end
    end
    Z_projected(:,:,1) = symm(Z_projected(:, :, 1)); % Ensure that the first block is symmetric
        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [val,X] = dual_gap(C, Z, gamma,nl,ng)
% computes the duality gap
    n = size(Z, 1);
    p = size(Z, 3) - 1;
    T_Z=get_T(Z);
    V = C + T_Z;

    W = get_W(C, Z);
    X = V\(sparse([W, zeros(n,n*p); zeros(n*p,n), zeros(n*p,n*p) ])/V);
    X = symm(X); % numerical accuracy

   val=-trace(X*T_Z)+get_hW(get_D_short(X, n ,p),gamma,nl,ng);

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function C=sTop(X,p)
% Computes the sample Toeplitz covariance matrix C of order p
% from the given data X (rows is time, columns are the component)
%
% Note that C has dimension n(p+1) x n(p+1) s.t.
%  
%       C=[C0  C1 .... Cn
%          C1' C0 .... : 
%          :           :
%          Cn'          ]
%
% where C_k=E[x(t+k)x(t)^T]
%




n=size(X,2);
N=size(X,1);
C_line=zeros(n,n,2*p+1);
for k=p+1:2*p+1
    for t=1:N-k+p+1
        C_line(:,:,k)=C_line(:,:,k)+N^-1*X(t+k-p-1,:)'*X(t,:);
    end
end
for k=1:p
    C_line(:,:,k)=C_line(:,:,-k+2*p+2)';
end
    
    

C = zeros(n*(p+1));
for i=1:p+1
    for j=1:p+1
        C(((i-1)*n+1):i*n,((j-1)*n+1):j*n)=C_line(:,:,(j-i)+p+1);
    end
end

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function T = get_T(Z)
% Function to compute T(Z), outputs a block Toeplitz matrix
    n = size(Z, 1);
    p = size(Z, 3) - 1;
    T = zeros(n*(p+1));
    for i=1:p+1
        for j=1:p+1
            if i<j
                T(((i-1)*n+1):i*n,((j-1)*n+1):j*n)=Z(:,:,(j-i)+1);
            else
                T(((i-1)*n+1):i*n,((j-1)*n+1):j*n)=(Z(:,:,(i-j)+1))';
            end
        end
    end
    T = symm(T); % It should be symmetric
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function D = get_D(X, n, p)
% Function to compute the adjoint of T with output in a matrix of size n x (p+1)    
    D = zeros(n,n*(p+1));
    %D0
    for l=0:p
        D(:,1:n)=D(:,1:n)+X(((l)*n+1):((l+1)*n),((l)*n+1):((l+1)*n));
    end
    
    %Dj for j in [1,p]
    for m=1:p
        for l=0:p-m
            D(:,(m*n+1:(m+1)*n)) = D(:,(m*n+1:(m+1)*n)) + 2*X(((l)*n+1):((l+1)*n),((l+m)*n+1):((l+m+1)*n));
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function D = get_D_short(X, n, p)
% Function to compute the adjoint of T with output in 3-d array
% p is the AR order
% n is the dimension of the process
    D = zeros(n,n,(p+1));
    %D0
    for l=0:p
        D(:,:,1)=D(:,:,1)+X(((l)*n+1):((l+1)*n),((l)*n+1):((l+1)*n));
    end
    %Dj for j in [1,p]
    for m=1:p
        for l=0:p-m
            D(:,:,m+1) = D(:,:,m+1) + 2*X(((l)*n+1):((l+1)*n),((l+m)*n+1):((l+m+1)*n));
        end
    end    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function boolean1 = check_positive_definiteness(C,Z)
% Checking one of the conditions for the backtracking step    
    
    T_Z = get_T(Z);
    
    
    n = size(Z, 1);
    
    V = C + T_Z;
    V = symm(V);
    
    % First part, positive semidefiniteness of V
    cond11 = min(eig(V)) >= -1e-8; % BM: earlier it was 0
    
    % Second part, positive semidefiniteness of the Schur complement
    V00 = V(1:n, 1:n);
    V1p0 = V(n+1 : end, 1:n);
    V1p1p = V(n+1 : end, n+1 : end);
    
    V_temp = V00 - V1p0'*(V1p1p\V1p0);
    V_temp = symm(V_temp); % BM
    cond12 = min(eig(V_temp))>= -1e-8; % BM: earlier it was 0
    
    % Synthesis
    boolean1 = cond12 && cond11;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function boolean1 = check_armijo(obj_old,C,Z,Zn,gradZ,alpha,t)
% Checking che armijo conditions
    
    p=size(Z,3)-1;
    val=0;
    for k=1:p+1
        val=val+trace(gradZ(:,:,k)'*(Zn(:,:,k)-Z(:,:,k)));
    end
    
    if  phi(C,Zn)<= obj_old+alpha*t*val
        boolean1=true;
    else 
        boolean1=false;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function W = get_W(C, Z)
% Function to compute W
    T_Z = get_T(Z);
    n = size(Z,1);
    V = C + T_Z;
    
    V00 = V(1:n,1:n);
    V1p0 = V(n+1:end,1:n);
    V1p1p = V(n+1:end,n+1:end);
    W = V00 -V1p0'*(V1p1p\V1p0);
    W = symm(W);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function D = symm(D)
% Function to compute the symmetric part of a matrix
    D = 0.5*(D + D');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [fit,reg] = fit_vs_reg(C, Z, gamma,nl,ng)
% computes the fit and the regularization part at the optimum 

    n = size(Z, 1);
    p = size(Z, 3) - 1;
    T_Z=get_T(Z);
    V = C + T_Z;

    W = get_W(C, Z);
    X = V\(sparse([W, zeros(n,n*p); zeros(n*p,n), zeros(n*p,n*p) ])/V);
    X = symm(X); % numerical accuracy

   fit = -log(det(X(1:n,1:n)))+trace(X*C);
   reg = get_hW(get_D_short(X, n ,p),gamma,nl,ng);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function d = distance(X,Y)
% computes the distance between two 3d matrices
p=size(X,3)-1;
n=size(X,1);
d=0;
for k=1:p+1
    d=d+trace((X(:,:,k)-Y(:,:,k))*(X(:,:,k)-Y(:,:,k))');
end
d= d/(n^2*(p+1));    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
 function [Lambda_new, Gamma_new]=reweight(S,Lambda_old,Gamma_old,T,e,nl,ng)

    Dabs = abs(S);
    Dabs_max_matrix = max(Dabs, [], 3); 
    for k=1:nl
        for j=k:nl
            W1 = Dabs_max_matrix((k-1)*ng+1:k*ng,(j-1)*ng+1:j*ng);
            W2 = Dabs_max_matrix((j-1)*ng+1:j*ng,(k-1)*ng+1:k*ng);
            W1 = triu(max(W1,W1'));
            W2 = triu(max(W2,W2'));
            W((k-1)*ng+1:k*ng,(j-1)*ng+1:j*ng)=max(W1,W2);
        end
    end
    Lambda_new=Lambda_old;
    gam = reshape(triu(Gamma_old),ng^2,1);
    [gams is]=sort(-gam);
    sol = -gams(1:ng*(ng+1)/2);
    for j=1:nl
        for k=j:nl
            Ttmp=T((j-1)*ng+1:ng*j,(k-1)*ng+1:ng*k);
            Wtmp =W((j-1)*ng+1:ng*j,(k-1)*ng+1:ng*k);
            v=reshape(Wtmp,ng^2,1);
            u=reshape(Ttmp,ng^2,1);
            v=v(is);
            u=u(is);
            for l=1:ng*(ng+1)/2
                sol(ng*(ng+1)/2+l,1)=sum(u(l:end))/(sum(v(l:end))+e);
            end
            for l=1:ng*(ng+1)
                obj(l,1)=obj_hyperLambda(S,T,Lambda_new,Gamma_old,e,sol(l),j,k);
            end
            if sum(sum(Gamma_old<1e-8))>0
                sol(ng*(ng+1)+1,1)=0;
                obj(ng*(ng+1)+1,1)=obj_hyperGamma(S,T,Lambda_new,Gamma_old,e,0,j,k);
            end
            [val pos]=min(obj);
            Lambda_new(k,j)=sol(pos);
            Lambda_new(j,k)=Lambda_new(k,j);
       end
    end
    clear sol pos obj
    Gamma_new=Gamma_old;
    lam = reshape(triu(Lambda_new),nl^2,1);
    [lams is]=sort(-lam);
    sol = -lams(1:nl*(nl+1)/2);
    for j=1:ng
        for k=j:ng
            Ttmp=T(j:ng:end,k:ng:end);
            Wtmp =W(j:ng:end,k:ng:end);
            v=reshape(Wtmp,nl^2,1);
            u=reshape(Ttmp,nl^2,1);
            v=v(is);
            u=u(is);
            for l=1:nl*(nl+1)/2
                sol(nl*(nl+1)/2+l,1)=sum(u(l:end))/(sum(v(l:end))+e);
            end
            for l=1:nl*(nl+1)
                obj(l,1)=obj_hyperGamma(S,T,Lambda_new,Gamma_new,e,sol(l),j,k);
            end
            if  sum(sum(Lambda_new<1e-8))>0 
                sol(nl*(nl+1)+1,1)=0;
                obj(nl*(nl+1)+1,1)=obj_hyperGamma(S,T,Lambda_new,Gamma_new,e,0,j,k);
            end
            [val pos]=min(obj);
            Gamma_new(k,j)=sol(pos);
            Gamma_new(j,k)=Gamma_new(k,j);
       end
    end
   
end
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function obj = obj_hyperGamma(S,T,Lambda,Gamma,e,x,j,k)

nl=size(Lambda,1);
ng=size(Gamma,1);
Gamma(j,k)=x;
Gamma(k,j) = Gamma(j,k);
WA =  max(kron(Lambda,ones(ng,ng)),kron(ones(nl,nl),Gamma)) ;  
obj =  get_hW(S,WA,nl,ng) - sum(sum(T.*log(WA)))+e*sum(sum(triu(Lambda)))+e*sum(sum(triu(Gamma)));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function obj = obj_hyperLambda(S,T,Lambda,Gamma,e,x,j,k)

nl=size(Lambda,1);
ng=size(Gamma,1);
Lambda(j,k)=x;
Lambda(k,j) = Lambda(j,k);
WA =  max(kron(Lambda,ones(ng,ng)),kron(ones(nl,nl),Gamma)) ;  
obj =  get_hW(S,WA,nl,ng) - sum(sum(T.*log(WA)))+e*sum(sum(triu(Lambda)))+e*sum(sum(triu(Gamma)));

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
