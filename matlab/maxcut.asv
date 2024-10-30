n = 100; % graph size
p = 3/n; % Erdos-Renyi edge probability
beta = 1; % reg paramater

N = ceil(beta); % number of stochastic matvecs
eta = 2/(beta); % step size
maxIter = ceil(100*beta); % number of iterations
Nfinal = ceil(beta^2); % number of stochastic matvecs in postprocessing step


C = randn(n,n);
C = C + C'; 


lambda = zeros(n,1);

lambdaHist = zeros(n,maxIter); % will store dual variable history
lambdaHist2 = zeros(n,maxIter); % will store history of running averages
dualHist = zeros(maxIter,1); % will store dual objective history
feasHist = zeros(maxIter,1); % will store primal feasibility history
tic
% main loop
for iter = 1:maxIter

    % effective cost matrix
    % A = C - spdiags(lambda,0,n,n); 
    A = C - diag(lambda);
    
    % diagonal
    B = expm(-beta*(A));
    tr = trace(B);
    a = diag(B)/tr;
    
    
    
    % gradient
    gr = ones(n,1)/n - a;

    % l infinity gradient ascent step    
    lambda = lambda + eta*norm(gr,1)*sign(gr);

    % project to zero-mean
    lambda = lambda-mean(lambda);

    % measure primal feasibility error (diagonal constraint error)
    feasHist(iter) = norm(a-(1/n),1);

    % measure dual objective
    dualHist(iter) = (1/beta)*log(tr);
    
    % store dual variable history
    lambdaHist(:,iter) = lambda;
    lambdaHist2(:,iter) = mean(lambdaHist(:,round(iter/2):iter),2);

end
optTime = toc;

disp(['Optim time: ',num2str(optTime),' seconds']);

figure(1);clf;plot(dualHist);title('Dual objective');
figure(2);clf;plot(feasHist);title('Primal feasibility');
