% Copyright   : Michael Pokojovy, Ebenezer Nkum and Thomas M. Fullerton, Jr. (2023)
% Version     : 1.0
% Last edited : 11/15/2023
% License     : Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
%               https://creativecommons.org/licenses/by-sa/4.0/

function [I_sigma_hat, alpha_hat, lambda_hat, var_rat] = HJM_inv_map(T_grid, X_grid, Y_obs, pca_var_pct, reg_eps)
    dx = X_grid(2) - X_grid(1);
    dt = T_grid(2) - T_grid(1);

    n = length(X_grid) - 2;
    e = ones(n, 1);

    D_D = (1/dx)*spdiags([-e e], [-1 0], n, n);
    D_N = (1/dx)*spdiags([-e e], [ 0 1], n, n);
    A = D_D\(D_N*D_D); % same as inv(D_D)*(D_N*D_D)

    % "sample" vectors
    pred = zeros(length(T_grid) - 1, n);
    
    for i = 2:length(T_grid)
        f1  = Y_obs(i,     2:end-1)';
        f0  = Y_obs(i - 1, 2:end-1)';

        % explicit Euler
        % Af0 = A*f0;

        % implicit Euler
        Af1 = A*f1;

        res = (f1 - f0)/dt - Af1;

        pred(i - 1, :) = res;
    end

    mu = mean(pred, 1);

    %% Estimate I_sigma
    S = cov(pred)*dt;
    [V, D] = eig(S);
    d = diag(D);

    var_rat = cumsum(flip(abs(d)))./sum(d);
    n_mode  = min(find(var_rat >= pca_var_pct));

    I = n:-1:(n - n_mode + 1);
    
    I_sigma_hat0 = spdiags(sqrt(d(I)), 0, n_mode, n_mode)*V(:, I)'; % Note the normalization factor dt

    S_proj  = I_sigma_hat0'*I_sigma_hat0;
    iS_proj = V(:, I)*diag(1.0./d(I))*V(:, I)'; % similar to pinv(S_proj)

    %% Estimate alpha (drift) and lambda (price of volatility)
    [I_sigma_hat, lambda_hat] = fit(I_sigma_hat0, n_mode);

    I_sigma_hat = I_sigma_hat';
    alpha_hat   = 0.5*sum(I_sigma_hat.^2, 1) - lambda_hat*I_sigma_hat;

    I = find(lambda_hat < 0);
    lambda_hat(I)     = -lambda_hat(I);
    I_sigma_hat(I, :) = -I_sigma_hat(I, :);

    %% Expand to boundaries
    %Extrapolation plus reduction of "boundary layer" effects
    I_sigma_hat = [zeros(n_mode, 1), I_sigma_hat(:, 1:end-1), ...
                   2*I_sigma_hat(:, end-1) - I_sigma_hat(:, end-2), ...
                   3*I_sigma_hat(:, end-1) - 2*I_sigma_hat(:, end-2)];

    alpha_hat = [0 alpha_hat(:, 1:end-1) 2*alpha_hat(:, end-1) - alpha_hat(:, end-2) ...
                 3*alpha_hat(:, end-1) - 2*alpha_hat(:, end-2)];

    function [I_sigma_hat, lambda_hat] = fit(I_sigma_hat0, n_mode)
        I_sigma_hat0 = I_sigma_hat0';
        
        n = size(I_sigma_hat0, 1);

        C0      = eye(n_mode);
        lambda0 = zeros(1, n_mode);

        x0 = reshape([C0; lambda0], n_mode + n_mode^2, 1);

        W      = spdiags(1./(X_grid(2:end - 1))', 0, n, n);
        S_norm = sum((W*S*W).^2, 'all');

        options = optimoptions('fmincon');
        options = optimoptions(options, 'MaxIter', n_mode*1E4, 'MaxFunEvals', n_mode*1E4);
        options = optimoptions(options, 'Algorithm', 'active-set');

        x = fmincon(@obj, x0, [], [], [], [], [], [], @nonlcon, options);

        x = reshape(x, n_mode + 1, n_mode);
        C = x(1:n_mode, :);
        lambda_hat  = x(n_mode + 1, :);
        I_sigma_hat = I_sigma_hat0*C;

        function res = obj(x)
            x = reshape(x, n_mode + 1, n_mode);

            C           = x(1:n_mode, :);
            lambda_hat  = x(n_mode + 1, :);
            I_sigma_hat = I_sigma_hat0*C;

            def = (0.5*sum(I_sigma_hat.^2, 2) - I_sigma_hat*lambda_hat' - mu')./(X_grid(2:end - 1)'); % defect

            res = sum(def.^2);
        end

        function [c, ceq] = nonlcon(x)
            x = reshape(x, n_mode + 1, n_mode);

            C           = x(1:n_mode, :);
            I_sigma_hat = I_sigma_hat0*C;

            Sigma = I_sigma_hat*I_sigma_hat';

            c = sum((W*(Sigma - S)*W).^2, 'all')/S_norm - reg_eps;

            ceq = [];
        end
    end
end