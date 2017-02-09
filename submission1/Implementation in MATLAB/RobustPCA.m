function [L, S] = RobustPCA(X, lambda, mu, tol, max_iter)
    % - X is a data matrix (of the size N x M) to be decomposed
    %   X can also contain NaN's for unobserved values
    % - lambda - regularization parameter, default = 1/sqrt(max(M,N))
    % - mu - the augmented lagrangian parameter, default = 10*lambda
    % - tol - reconstruction error tolerance, default = 10^(-7).
    % - max_iter - maximum number of iterations, default = 1000.

    [M, N] = size(X);
    unobserved = isnan(X);      % If the entries are Not a Number(NaN) then we will consider them to be unobserved.
    X(unobserved) = 0;          % Initializing the unobserved entries to be zero.
    normX = norm(X, 'fro');     % Calculating the Frobenius norm.

    % Default arguments
    if nargin < 2
        lambda = 1/sqrt(max(size(X)));
    end
    if nargin < 3
        mu = 10*lambda;
    end
    if nargin < 4
        tol = 1e-7;
    end
    if nargin < 5
        max_iter = 1000;
    end
    
    % Initial solution
    L = zeros(M, N);
    S = zeros(M, N);
    Y = zeros(M, N);
    
    for iter = (1:max_iter)
        % Updating L and S
        L = Do(1/mu, X - S + (1/mu)*Y);      % Using Singular Value thresholding.
        S = So(lambda/mu, X - L + (1/mu)*Y); % Using the Shrinkage operator.
        Z = X - L - S;                       % Augmented lagrangian multiplier
        Z(unobserved) = 0;                   % skipping the unobserved values.
        Y = Y + mu*Z;                        % Updating the Lagrange Multiplier.
        
        err = norm(Z, 'fro') / normX;        % Calculating the error.
        
        % Checking the terminating conditions.
        if (iter == 1) || (mod(iter, 10) == 0) || (err < tol)
            fprintf(1, 'iter: %04d\terr: %f\trank(L): %d\tcard(S): %d\n', ...
                    iter, err, rank(L), nnz(S(~unobserved)));
        end
        if (err < tol) break; end
    end
end

function r = So(tau, X)
    % Shrinkage operator
    r = sign(X) .* max(abs(X) - tau, 0);
end

function r = Do(tau, X)
    % Singular value thresholding.
    [U, S, V] = svd(X, 'econ');
    r = U*So(tau, S)*V';
end
