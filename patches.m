function [PM,num_patches] = patches(M,patch_size)
% Take the patches of a sample across the spatial dimensions
% Input:
%       M           : Original tensor (N-way array)
%       patch_size  : Spatial size of each patch (2 x 1 vector) 
%
% Output:
%       PM          : Patches of the tensor (cell array)
%       num_patches : Number of patches (scalar)
%

dim = size(M);
N = ndims(M);
n1 = ceil(dim(1)/patch_size(1));
n2 = ceil(dim(2)/patch_size(2));
num_patches = n1*n2;
PM = cell(num_patches,1);
k = 0;
for i = 1:n1
    % Calculate the rows of each patch in the initial tensor
    if i == n1
        x = (i-1)*patch_size(1)+1:dim(1);
    else
        x = (i-1)*patch_size(1)+1:i*patch_size(1);
    end
    for j = 1:n2
        k = k+1;
        % Calculate the columns of each patch in the initial tensor
        if j == n2
            y = (j-1)*patch_size(2)+1:dim(2);
        else
            y = (j-1)*patch_size(2)+1:j*patch_size(2);
        end
        % Take the patches
        if N == 2
            PM{k} = M(x,y);
        elseif N == 3
            PM{k} = M(x,y,:);
        elseif N == 4
            PM{k} = M(x,y,:,:);
        elseif N == 5
            PM{k} = M(x,y,:,:,:);
        else
            error('The tensor are not 2D, or 3D or 4D or 5D.\n');
        end
    end
end
end