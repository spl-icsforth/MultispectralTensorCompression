function Mrec = reconstruction(Grec,PDnew,patch_size,RR,dim)
% Reconstruct the samples from the received data using the learned basis matrices
% Inputs:
%        Grec       : Core tensor of the testing sample (N-way tensor)
%        PDnew      : Learned basis matrices (N x 1 cell array)
%        patch_size : Spatial size of each patch (2 x 1 vector)
%        RR         : Dimensions of the core tensor as percentage of the 
%                     original dimensions (1 x N vector)
%        dim        : Size of the samples (N x 1 vector)
%
% Outputs:
%        Mrec : Reconstructed testing sample (N-way tensor)
%

N = ndims(Grec);

if N == 1
    Mrec = PDnew{1}{1}*Grec;
else
    [GG,num_patches] = patches(Grec,round(patch_size.*RR(1:2)));
    PMrec = cell(num_patches,1);
    for k = 1:num_patches
        % Compute the reconstructed testing sample for each patch
        if ndims(GG{k})<N
            mm = zeros(1,N);
            for n = 1:N
                mm(n) = size(PDnew{k}{n},2);
            end
            PMrec{k} = GG{k};
            for n = 1:N
                g = PDnew{k}{n}*Unfold(PMrec{k},mm,n);
                mm(n) = size(PDnew{k}{n},1);
                PMrec{k} = Fold(g,mm,n);
            end
        else
            PMrec{k} = double(ttm(GG{k},PDnew{k}));
        end
    end
    Mrec = union_patches(PMrec,dim,patch_size);
end
end