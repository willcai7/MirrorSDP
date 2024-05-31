rng(1)

% marginal sizes
m = 100;
n = 90;

% inverse temp
beta = 30;

eta = .5/beta; % step size
maxIter = ceil(800*beta); % number of iterations

%{
% sample random points with uniform marginals
x = sort(randn(m,1));
y = sort(randn(n,1));
mu = ones(m,1)/m;
nu = ones(n,1)/n;
%}

% equispaced points with gaussian / double-well marginals
x = linspace(-4,4,m)';
y = linspace(-4,4,n)';
mu = exp(-x.^2/2);
nu = exp(-(y.^4-y.^2)/2);
mu = mu/sum(mu);
nu = nu/sum(nu);

% form quadratic cost
C = (x-y.').^2;

% initialize dual variables
phi = zeros(m,1);
psi = zeros(n,1);

% prepare to launch
dualHist = zeros(maxIter,1);
feasHist = zeros(maxIter,1);

% main loop
for iter=1:maxIter
    A = C - phi - psi';
    X = exp(-beta*A);
    val = sum(sum(X));
    dualHist(iter) = dot(phi,mu)+dot(psi,nu)-(1/beta)*log(val);
    P = X/val;
    gr1 = mu - sum(P,2);
    gr2 = nu - sum(P,1)';
    feasHist(iter) = norm(gr1,1)+norm(gr2,1);
    phi = phi + eta*norm(gr1,1)*sign(gr1);
    psi = psi + eta*norm(gr2,1)*sign(gr2);
end

figure(1);plot(dualHist);title('Dual objective')
figure(2);semilogy(feasHist);title('Primal feasibility')
figure(3);imagesc(P);title('Pair marginal')