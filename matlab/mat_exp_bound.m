C = randn(10);
C = C + C';
lambda = randn(10,1);
res = zeros(1000,1);
for i=1:1000
    lambda(1) = i-500;
    A = diag(lambda) - C;
    B = diag(expm(A));
    res(i) = B(1,1);
end
subplot(1,2,1)
lambdas = (1:1000)-500; 
semilogy(lambdas, res);

subplot(1,2,2)
plot(lambdas, res);
