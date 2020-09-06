function flag = check_orthogonality(num_of_orthogonal, factor_cell)

for i = length(factor_cell) - num_of_orthogonal +1  : length(factor_cell)
    if norm(factor_cell{i}'* factor_cell{i} - eye(size(factor_cell{i},2 ))) < 1e-6
        flag = 1;
        return;
    else
        flag = 0;
        return;
    end    
end

    flag = 0;
    return;