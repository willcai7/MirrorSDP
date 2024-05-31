rng(1)

% marginal sizes
m = 200;
n = 190;

% inverse temp
beta = 1000;

eta = .5/beta; % step size
maxIter = ceil(10*(beta)); % number of iterations

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
C = 1./sqrt(.1^2 + (x-y.').^2 );

% initialize dual variables
phi = zeros(m,1);
psi = zeros(n,1);

% prepare to launch
dualHist = zeros(maxIter,1);
feasHist = zeros(maxIter,1);

tprev = 1;
phiprev = phi;
psiprev = psi;

% main loop
for iter=1:maxIter
    t = (1+sqrt(1+4*tprev^2))/2;

    A = C - phi - psi';
    X = exp(-beta*A);
    val = sum(sum(X));
    dualHist(iter) = dot(phi,mu)+dot(psi,nu)-(1/beta)*log(val);
    P = X/val;

    muhat = sum(P,2);
    nuhat = sum(P,1)';
    gr1 = mu - muhat;
    gr2 = nu - nuhat;
    feasHist(iter) = norm(gr1,1)+norm(gr2,1);

    phi = phi + (1/beta)*(log(mu)-log(muhat));
    psi = psi + (1/beta)*(log(nu)-log(nuhat));
end

figure(1);plot(dualHist);title('Dual objective')
figure(2);semilogy(feasHist);title('Primal feasibility')
figure(3);imagesc(P);title('Pair marginal')