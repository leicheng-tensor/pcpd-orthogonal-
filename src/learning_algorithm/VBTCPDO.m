function learning_results = VBTCPDO(Y, num_of_orthogonal )
%%  Initializations 
max_iteraton = 100; 
conv_thre = 1e-6;  % this could be changed for different applications 
rank_ratio = 20;  % this could be changed for different applications 
Y_max = max(Y(:));
if Y_max > 10    % this could be changed for different applications
    outlier_thre = .5;  % this could be changed for different applications
else 
    outlier_thre = Inf;
end    
sizeY = size(Y); 
num_of_factors = length(sizeY);
R   = max(sizeY); % R = min(sizeY) if R is known to be less than the min(sizeY)
Y = tensor(Y);
a_gamma_0 = 1e-6;
b_gamma_0 = 1e-6;
c_xi_0 = 1e-6;
d_xi_0  = 1e-6;
e_beta_0 = 1e-6;
f_beta_0 = 1e-6;
gammas = (a_gamma_0/b_gamma_0) * ones(R,1);
xis = (c_xi_0/d_xi_0)*ones(sizeY);
beta = e_beta_0/f_beta_0;
Sigma_cell = cell(num_of_factors,1);
for k = 1:num_of_factors
    Sigma_cell{k} = eye(R);
end    
factor_cell = cell(num_of_factors,1);

if num_of_orthogonal == length(sizeY)
    disp('Warning: The number of orthogonal factors must be smaller than the number of factor matrices !');
    learning_results = '';
    return;
end    

if num_of_orthogonal ~= 0
    for k = 1:num_of_factors
       [temp, ~, ~] = svd(double(tenmat(Y,k)), 0); 
       if R <= size(temp,2)
       factor_cell{k} = temp(:,1:R);
       else 
       factor_cell{k} = [temp, random(size(temp,1), R-size(temp,2))];
       end
    end
end

if num_of_orthogonal == 0
    for k = 1:num_of_factors
       [tempU, tempS, ~] = svd(double(tenmat(Y,k)), 0); 
       if R <= size(tempU,2)
       factor_cell{k} = tempU(:,1:R)*(tempS(1:R,1:R).^(0.5) ) ;
       else 
       factor_cell{k} = [tempU, randn(size(tempU,1), R-size(tempU,2))];
       end
    end
end

X= conj(double(ktensor(factor_cell)));

temp_Hproducts = cell(num_of_factors-num_of_orthogonal,1);
for n= 1:(num_of_factors-num_of_orthogonal)
    temp_Hproducts{n} = factor_cell{n}'*factor_cell{n}+ sizeY(n)*Sigma_cell{n};
end

%% Iterative algorithms to learn model parameters 
for it=1:max_iteraton,
    %% Update the Outliers
    Sigma_E = 1./(xis+beta);
    E = double(beta*(Y-X).*Sigma_E);
   
    for k= 1:prod(sizeY)
        if abs(E(k))^2<=outlier_thre
            E(k)=0;
        end
    end
    
    %% Update the xis
    c_xi_it = c_xi_0 + 1;
    d_xi_it = d_xi_0 + abs(E).^2 + Sigma_E;
    xis = c_xi_it./d_xi_it;
    
    %% Update factor matrices with no constraints
    Lamba_w = diag(gammas);
    
    if num_of_orthogonal ~= 0
        for n = 1:(num_of_factors-num_of_orthogonal)
            tempit_Hproducts = ones(R,R);
            for m = [1:n-1, n+1:(num_of_factors-num_of_orthogonal)]
                tempit_Hproducts =  tempit_Hproducts.*temp_Hproducts{m};
            end
            data_differ =double(tenmat((Y-E), n));
            temp_data = double(khatrirao_fast(factor_cell{[1:n-1, n+1:num_of_factors]},'r')' *data_differ.');
            Sigma_cell{n} = (beta * diag(diag(tempit_Hproducts)) + Lamba_w)^(-1);
            factor_cell{n} = (beta * Sigma_cell{n} * temp_data).';
            temp_Hproducts{n} = factor_cell{n}'*factor_cell{n} + sizeY(n)*Sigma_cell{n};
        end
    end
    
    if num_of_orthogonal == 0
        for n = 1:(num_of_factors-num_of_orthogonal)
            tempit_Hproducts = ones(R,R);
            for m = [1:n-1, n+1:(num_of_factors-num_of_orthogonal)]
                tempit_Hproducts =  tempit_Hproducts.*temp_Hproducts{m};
            end
            data_differ =double(tenmat((Y-E), n));
            temp_data = double(khatrirao_fast(factor_cell{[1:n-1, n+1:num_of_factors]},'r')' *data_differ.');
            Sigma_cell{n} = (beta * tempit_Hproducts + Lamba_w)^(-1);
            factor_cell{n} = (beta * Sigma_cell{n} * temp_data).';
            temp_Hproducts{n} = factor_cell{n}'*factor_cell{n} + sizeY(n)*Sigma_cell{n};
        end
    end
    
    
    %% Update factor matrices with orthogonal constraints
    for n = (num_of_factors-num_of_orthogonal+1) : num_of_factors
        data_differ=double(tenmat((Y-E), n));
        data_differ_temp = double(khatrirao_fast(factor_cell{[1:n-1, n+1:num_of_factors]},'r')' *data_differ.');
        [Us,~,Vs]=svd((beta * data_differ_temp).',0);
        factor_cell{n}=Us(:,1:R)*Vs(:,1:R)';
    end
    %% Update latent tensor X
    diff_conv =conj(double(ktensor(factor_cell))) - X;
    if norm(diff_conv(:),'fro')^2<=conv_thre
        break;
    end
    X= conj(double(ktensor(factor_cell)));
    %% Update hyperparameters gamma
    a_gamma_it = (sum(sizeY(1:num_of_factors-num_of_orthogonal)) + a_gamma_0)*ones(R,1);
    b_gamma_temp = 0;
    for n = 1:num_of_factors - num_of_orthogonal
        b_gamma_temp = b_gamma_temp + diag(factor_cell{n}'*factor_cell{n}) + sizeY(n)*diag(Sigma_cell{n});
    end
    b_gamma_it = b_gamma_0 + b_gamma_temp;
    gammas = a_gamma_it./b_gamma_it;
    
    %% update noise beta
    f_temp1 = ones(R,R);
    
    if num_of_orthogonal ~= 0
        for n=1:num_of_factors - num_of_orthogonal
            f_temp1 = f_temp1.*temp_Hproducts{n};
        end
        f_temp1 = sum(diag(conj(f_temp1)));
    end
    
    if num_of_orthogonal == 0
        for n=2:num_of_factors - num_of_orthogonal
            f_temp1 = f_temp1.*temp_Hproducts{n};
        end
        f_temp1 = sum(diag(  temp_Hproducts{1}*conj(f_temp1)));
    end    
    
    f_temp2 = sum((abs(E(:)).^2 + Sigma_E(:)));
    f_temp = Y(:)'*Y(:) - 2*real(Y(:)'*X(:)) -2*real(Y(:)'*E(:)) + 2*real(X(:)'*E(:)) + f_temp1 + f_temp2;
    e_beta_it = e_beta_0 + prod(sizeY);
    f_beta_it = f_beta_0 + f_temp;
    beta = e_beta_it/f_beta_it;  
    
    %% Automatic Rank Determination 
    rank_thre = min(gammas)*rank_ratio;
    if sum(find(gammas > rank_thre))
        selection = gammas <= rank_thre;
        for n=1:num_of_factors-num_of_orthogonal
            factor_cell{n} = factor_cell{n}(:,selection);
            Sigma_cell{n} = Sigma_cell{n}(selection,selection);
            temp_Hproducts{n} = temp_Hproducts{n}(selection,selection);
        end
    
    for n= num_of_factors-num_of_orthogonal+1:num_of_factors
        factor_cell{n}=factor_cell{n}(:,selection);
    end
    gammas = gammas(selection);
    R=length(gammas);
    end
   
    
    
end
%% Output
learning_results.X=X;
learning_results.factor_cell = factor_cell;
learning_results.E = E;
learning_results.R = R;