% Code written by Mattia Zorzi 2019
% Please cite the corresponding paper [Mattia Zorzi. "Autoregressive 
% Identification of Kronecker Graphical Models", 2020]
Install_mex
% algorithm parameters  
e = 10^-3; % hyperprior for Lambda and Gamma
tol=10^-3; % threshold for the reweightingò-like scheme
verb=' '; % no verbosity

% load data
load dataset   % nl=3 number of modules
               % ng=3 number of nodes in a module
               % p=2  order of the AR model
               % S    contains the coefficients of the inverse PSD
               % Omega is the support of the inverse PSD
               % y are the data
               
[Omega_est,S_est] = identKron8(y,p,nl,ng,e,tol,verb); % Omega_est is the estmated Omega
                                                      % S_est estimated coefficients of the inverse PSD                                                   
[Omega_est,S_est] = identKron8X(y,p,nl,ng,e,tol,verb);                                                         
    
   
 