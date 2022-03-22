function Grec = Estimate_core(test,PDD,patch_size,RR)
% Estimate the core tensor of the samples using the learned basis matrices
% Inputs:
%        test       : Testing sample (N-way tensor)
%        PDD        : Inverted learned basis matrices (N x 1 cell array)
%        patch_size : Spatial size of each patch (2 x 1 vector)
%        RR         : Dimensions of the core tensor as percentage of the 
%                     original dimensions (1 x N vector)
%
% Outputs:
%        Grec   : Core tensor of the testing sample (N-way tensor)
%

N = ndims(test);
dim = size(test);

if N == 1
    Grec = PDD{1}{1}*test;
    num = prod(size(Grec));
    cr = num/prod(size(test)); % Compression Ratio
else
    % Split the testing sample into patches
    [PMtest,num_patches] = patches(test,patch_size);
    
    num = 0;
    GG = cell(num_patches,1);
    for k = 1:num_patches
        % Estimate the core tensor for each patch
        GG{k} = double(ttm(PMtest{k},PDD{k}));
        num = num+prod(size(GG{k}));
    end
    Grec = union_patches(GG,round(RR.*dim),round(RR(1:2).*patch_size));
    cr = num/prod(size(test));
end
fprintf('Compression Ratio %2f, ',1-cr)
end