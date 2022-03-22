function [PDnew, PDD, Grec, Xrec, er] = Tensor_Decomposition_Learning(Xtrain,RR,patch_size)
% Learn the basis matrices of each dimension from training samples
% Inputs:
%        Xtrain     : (N+1)-way tensor that contains the N-way training samples
%                     (the last dimension indicates the samples)
%        RR         : 1 x (N+1) vector of the dimensions of the core tensor as
%                     percentage of the original dimensions
%        patch_size : 2 x 1 vector of the spatial size of each patch
%
% Outputs:
%        PDnew : Learned basis matrices (N x 1 cell array)
%        PDD   : Inverted learned basis matrices (N x 1 cell array)
%        Grec  : (N+1)-way tensor that contains the core tensors of the samples
%        Xrec  : (N+1)-way tensor that contains the reconstructed training tensors
%        er    : num_patches x 1 cell array of the NRMSE of the patches at
%                each iteration
%

N = ndims(Xtrain)-1;
dim = size(Xtrain);

% Parameters
itter = 50; % number of maximum iterations
p = 0.01;   % step-size parameter
tol = 1e-7; % tolerance for stopping criterion

% Training Process - ADMM - min L(G,D{1},..,D{N},A{1},..,A{N},Y{1},..,Y{N})
% iteratively, with respect to each variable
fprintf('Learn the basis matrices of each dimension from training samples\n')

% Split Xtrain into patches
[PXtrain, num_patches] = patches(Xtrain,patch_size);

PDnew = cell(num_patches,1);
PDD = cell(num_patches,1);
PMrec = cell(num_patches,1);
er = cell(num_patches,1);
Gnew = cell(num_patches,1);
r = zeros(num_patches,1);
for k = 1:num_patches
    PDnew{k} = cell(N,1);
    PDD{k} = cell(N,1);
%     fprintf('Patch %d/%d\n',k,num_patches)
    pdim = size(PXtrain{k});
    R = round(RR.*pdim);
    % Initialization of the variables
    D = cell(N+1,1); % Factor matrices for each dimension
    for n = 1:(N+1)
        if R(n) == 0
            R(n) = 1;
        end
        D{n} = rand(pdim(n),R(n));
        if n ~= (N+1) % Orthogonality constraints
            [Q,~] = qr(D{n});
            D{n} = Q(:,1:R(n));
        end
    end
    DD = cell(N+1,1); % Inverse factor matrices
    for n = 1:N
        DD{n} = D{n}';
    end
    DD{N+1} = pinv(D{N+1});
    A = D{N+1};                     % Auxiliary variable
    G = double(ttm(PXtrain{k},DD)); % Core tensor
    Y = zeros(size(D{N+1}));        % Lagrange Multiplier Matrix

    er{k} = [];
    for i = 1:itter
%         fprintf('Iteration: %d\n',i)
        % Update A (with the low-rank constraint)
        A = D{N+1}-(1/p)*Y;
        A = normc(A);
        [U1,S1,V1] = svd(A);
        s1 = diag(S1);
        ss = cumsum(s1)/sum(s1);
        rr = length(find(ss(:)<0.90));
        if i == 1
            r(k) = rr;
        end
        if rr == 0
            r(k) = 1;
        elseif rr < r(k)
            r(k) = rr;
        end
        A = U1(:,1:r(k))*S1(1:r(k),1:r(k))*V1(:,1:r(k))';
        % Update D{1},..,D{N},D{N+1}
        Dnew = cell(N+1,1);
        for n = 1:(N+1)
            CN = double(ttm(G,D,-n));
            Cnn = Unfold(CN,size(CN),n);
            if n == (N+1)
                Dnew{n} = (Unfold(PXtrain{k},size(PXtrain{k}),n)*Cnn'+Y+p*A)*pinv(Cnn*Cnn'+p*eye(R(n)));
                Dnew{n} = normc(Dnew{n});
            else
                Dnew{n} = (Unfold(PXtrain{k},size(PXtrain{k}),n)*Cnn')*pinv(Cnn*Cnn');
                [Q,~] = qr(Dnew{n});
                Dnew{n} = Q(:,1:R(n));
            end
        end
        D = Dnew;
        % Update G
        for n = 1:N
            DD{n} = D{n}';
        end
        DD{N+1} = pinv(D{N+1});
        G = double(ttm(PXtrain{k},DD));
        % Update Y
        Y = Y+p*(A-D{N+1});

        % Stopping criterion
        PMrec{k} = double(ttm(G,D));
        if isequal(PMrec{k},PXtrain{k})
            er{k}(i) = 0;
        else
            er{k}(i) = norm(PMrec{k}(:)-PXtrain{k}(:))/norm(PXtrain{k}(:));
        end
        if (i>1)&&(er{k}(i)>er{k}(i-1))
            break;
        else
            for n = 1:N
                PDnew{k}{n} = D{n};
                PDD{k}{n} = DD{n};
            end
            if (er{k}(i) <= tol) || ((i>1)&&(abs(er{k}(i)-er{k}(i-1))<=1e-4))
                break;
            end
        end
        
    end
    Gnew{k} = double(ttm(PXtrain{k},PDD{k},1:N));
end
% Estimate the reconstructed training tensor
Xrec = union_patches(PMrec,size(Xtrain),patch_size);
Grec = union_patches(Gnew,round(RR.*dim),round(RR(1:2).*patch_size));
err = norm(Xrec(:)-Xtrain(:))/norm(Xtrain(:));
fprintf('Training NRMSE of the Tensor Decomposition Learning process: %e\n',err)
end