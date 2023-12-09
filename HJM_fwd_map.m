% Copyright   : Michael Pokojovy, Ebenezer Nkum and Thomas M. Fullerton, Jr. (2023)
% Version     : 1.0
% Last edited : 11/15/2023
% License     : Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
%               https://creativecommons.org/licenses/by-sa/4.0/

function sol = HJM_fwd_map(T_grid, X_grid, Y0, I_sigma, alpha, n_rep)
    k = size(I_sigma, 1);
    
    dt = T_grid(2) - T_grid(1);
    dx = X_grid(2) - X_grid(1);
    
    n = length(X_grid) - 2;
    e = ones(n, 1);

    D_D = (1/dx)*spdiags([-e e], [-1 0], n, n);
    D_N = (1/dx)*spdiags([-e e], [ 0 1], n, n);
    A = D_D\(D_N*D_D);

    A_res = speye(n) - A*dt; % discrete semigroup via resolvent
    mu = alpha; % drift

    sol = zeros(length(T_grid), length(X_grid), n_rep);
    
    for j = 1:n_rep
        sol(1, :, j) = Y0;
        
        for i = 2:length(T_grid)
            dW = sqrt(dt)*randn(1, k);
            
            % Solve with (semi-)implicit Euler-Maruyama
            rhs = sol(i - 1, :, j) + dt*mu + dW*I_sigma;
            rhs = rhs';
            
            v = (A_res\rhs(2:end - 1));
            v = [0; v; (2*v(end) - v(end - 1))];
            sol(i, :, j) = v';
        end
    end
end