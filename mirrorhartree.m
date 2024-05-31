% grid size
n = 100;

% construct grid
h = 1/n;
xg = linspace(0,1,n+1);
xg = xg(1:end-1);

% specify potential (periodic double Coulomb well)
a = .3;
V = -1*diag( sqrt( 1./((cos(4*pi*xg)+1) + a^2) ) );

% construct Laplacian
L = zeros(n,n);
for i=1:n-1
    L(i,i) = 2;
    L(i,i+1) = -1;
    L(i+1,i) = -1;
end
L(n,n) = 2;
L(1,n) = -1;
L(n,1) = -1;
L = .01 * L/(h^2);

% construct fake coulomb kernel
K = 2*(1-sqrt(1-cos(2*pi*(xg'-xg))));

% construct noninteracting Hamiltonian
H0 = 0.5*L + V;
H0 = (H0+H0')/2;

% parameters
beta = 50; % inverse temp
maxIter = 10*beta;
N = 2; % number of electrons
alpha = 1/beta; % primal step size
gamma = 1/(2*beta); % dual step size
S = 50; % number of stochastic matvecs

% initialize
mu = 0; % chemical potential
Heff = H0; % effective hamiltonian
NhatHist = zeros(maxIter,1); % history of estimated electron numbers
rhoHist = zeros(n,maxIter); % history of estimated electron densities

% run
for iter=1:maxIter

    % do randomized matvecs by sqrt(P)
    [U,E] = eig(Heff);
    e = diag(E);
    e2 = (1./(1+exp(beta*e)));
    Phalf = U*sqrt(diag(e2))*U';
    Z = randn(n,S);
    W = Phalf*Z;

    % build electron density
    rho = sum(W.*W,2)/S;
    Nhat = sum(rho);

    % build hartree
    VH = K*rho;

    % dual ascent step for chemical potential
    if mod(iter,1)==0
        mu = mu - gamma*(Nhat-N);
    end

    % effective hamiltonian update (primal mirror descent step)
    Heff = (1-alpha)*Heff + alpha*(H0 - mu*eye(n) + diag(VH));
    
    % save history
    rhoHist(:,iter) = rho;
    NhatHist(iter) = Nhat;

end

figure(1);plot(NhatHist(2:end))
figure(2);plot(mean(rhoHist(:,maxIter/2:end),2))