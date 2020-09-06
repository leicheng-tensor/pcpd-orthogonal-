function [X,factor_cell] = tensor_generation(dim_list, num_of_orthogonal,tensor_rank)
% Return tensor X yeids a tensor CPD with factor matrices sampled from
% standard normal distribution. The tensor rank is R, and num_of_orthogonal
% factors are orthogonal. 
% Return factor_cell with each cell element being factor matrices.
% author: Lei Cheng 

factor_cell = cell(length(dim_list),1);

for i = 1 : length(dim_list) - num_of_orthogonal
    temp_factor =  sqrt(.5)*(  randn(dim_list(i),tensor_rank)...
                                +1j*randn(dim_list(i),tensor_rank));
    factor_cell{i} = temp_factor;
end   

for i = length(dim_list) - num_of_orthogonal +1  : length(dim_list) 
    temp_factor = sqrt(.5)*(  randn(dim_list(i),tensor_rank)...
                                +1j*randn(dim_list(i),tensor_rank));
    [U_temp,~,~]=svd(temp_factor,0);  % here the enconimic SVD is used
    factor_cell{i} = U_temp;    
end    

X = conj(double(ktensor(factor_cell)));
disp('Tensor is genenrated !');
disp('-------------------------------------');
end