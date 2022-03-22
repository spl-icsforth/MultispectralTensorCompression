function M = union_patches(PM,dim,patch_size)
% Stitch the patches to create the sample
% Input:
%       PM         : Patches of the tensor (cell array)
%       dim        : Dimensions of the original tensor (N x 1 vector)
%       patch_size : Spatial size of each patch (2 x 1 vector)
%
% Output:
%       M          : Unioned tensor (N-way array)
%

N = length(dim);
n1 = ceil(dim(1)/patch_size(1));
n2 = ceil(dim(2)/patch_size(2));

M = zeros(dim);
k = 0;
for i = 1:n1
    for j = 1:n2
        k = k+1;
        % Calculate the rows of each patch in the initial tensor
        if i == n1
            x = (i-1)*patch_size(1)+1:dim(1);
        else
            x = (i-1)*patch_size(1)+1:i*patch_size(1);
        end
        % Calculate the columns of each patch in the initial tensor
        if j == n2
            y = (j-1)*patch_size(2)+1:dim(2);
        else
            y = (j-1)*patch_size(2)+1:j*patch_size(2);
        end
        % Create the tensor from the patches
        if N == 2
            M(x,y) = PM{k};
        elseif N == 3
            M(x,y,:) = PM{k};
        elseif N == 4
            M(x,y,:,:) = PM{k};
        elseif N == 5
            M(x,y,:,:,:) = PM{k};
        else
            error('The tensor are not 2D, or 3D or 4D or 5D.\n');
        end
    end
end
end