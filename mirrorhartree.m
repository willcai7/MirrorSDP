%% Initialization 
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
ffermi = @(x) 1./(1+exp(beta*x)); % Fermi-Dirac function
sffermi = @(x) sum(1./(1+exp(beta*x))); % sum of Fermi-Dirac function


%% Mirror Descent 

% initialize
mu = 0; % chemical potential
Heff = H0; % effective hamiltonian
MD_NhatHist = zeros(maxIter,1); % history of estimated electron numbers
MD_rhoHist = zeros(n,maxIter); % history of estimated electron densities
MD_muHist = zeros(maxIter,1); % history of the chemical potentials
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
    MD_rhoHist(:,iter) = rho;
    MD_NhatHist(iter) = Nhat;
    MD_muHist(iter) = mu;

end



%% SCF 

range_mu=[-10000, 10000]; % chemical potential
Heff = H0; % effective hamiltonian
SCF_NhatHist = zeros(maxIter,1); % history of estimated electron numbers
SCF_rhoHist = zeros(n,maxIter); % history of estimated electron densities
SCF_muHist = zeros(maxIter,1); % history of the chemical potentials
precision = 1e-6;

for iter =1:maxIter 
    [U,E] = eig(Heff);
    e = diag(E);

    % Find best mu 
    mu_lb = range_mu(1);
    mu_ub = range_mu(2);
    mu_mid = (mu_lb + mu_ub)/2;
    while (abs(sffermi(e-mu_mid)-N) >precision)
        if sffermi(e-mu_mid)-N>0
            mu_ub = mu_mid;
            mu_mid = (mu_lb + mu_ub)/2;
        else
            mu_lb = mu_mid;
            mu_mid = (mu_lb + mu_ub)/2;
        end
    end
    
    % Get rho by the FD function
    rho = diag(U*diag(ffermi(e-mu_mid))*U');
    Nhat = sum(rho);

    % Build Hartree
    VH = K*rho;

    % effective hamiltonian update (primal mirror descent step)
    Heff = H0 + diag(VH);
    
    % save history
    SCF_rhoHist(:,iter) = rho;
    SCF_NhatHist(iter) = Nhat;
    SCF_muHist(iter) = mu;

end

%% Plot 

figure(1);
subplot(1,3,2);
plot(MD_NhatHist(2:end));
hold on;
plot(SCF_NhatHist(2:end));
xlabel("Iterations");
title("$\hat N$", 'Interpreter','latex')

subplot(1,3,1);
plot(mean(MD_rhoHist(:,maxIter/2:end),2));
hold on;
plot(mean(SCF_rhoHist(:,maxIter/2:end),2));
title("Average $\rho$", 'Interpreter','latex')

subplot(1,3,3);
plot(MD_muHist(2:end));
hold on;
plot(SCF_muHist(2:end));
legend("Mirror Descent", "Deterministic SCF");
title("$\mu$", 'Interpreter','latex')
xlabel("Iterations");