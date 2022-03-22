clear
close all
clc

addpath('tensor_toolbox')
rand('seed',2018);

% Parameters
num_train = 50;     % number of training samples
num_test = 50;      % number of testing samples
RR = [0.5 0.5 0.5]; % percentage of the original dimensions for the size of the core tensor
patch_size = [8 8]; % size of spatial dimensions for each patch
bit = 8;            % number of quantization bits

% Load the data
load('Data/NASA2017_Chania.mat')
Xtrain = M(:,:,:,1:num_train); % Training set
load('Data/NASA2018_Chania.mat')
Xtest = M(:,:,:,1:num_test);   % Testing set
clear('M');

N = ndims(Xtrain)-1; % Number of dimensions of each sample (3D case)
dim = size(Xtrain);
dim = dim(1:N);      % Size of each sample

%% Training Process
fprintf('Training Process\n')
% Tensor Decomposition Learning
[PDnew, PDD, Grec_train, Xrec, er] = Tensor_Decomposition_Learning(Xtrain,[RR 1],patch_size);

% Learn the symbols and the dictionary for Huffman coding using Lloyd-max quantization
fprintf('Learn the symbols and the dictionary for Huffman coding\n')
% Lloyd-max quantization of the coefficients
gg = unique(Grec_train(:));
idx = 2:20:length(gg);
uu = gg(idx);
[lq,symbols,~] = lloyd_max(uu,bit,min(gg),max(gg));
% Learn the dictionary for Huffman coding
prob = zeros(length(symbols),1);
for j = 1:length(symbols)
    prob(j) = length(find(lq==j));
end
prob = prob./sum(prob);
dict = huffmandict(symbols,prob); % Dictionary - Huffman coding

%% Testing Process
fprintf('Testing Process\n')

Grec = cell(num_test,1);
GG_quant = cell(num_test,1);
Mrec = cell(num_test,1);
bpppb = zeros(num_test,1);
bits_coding = zeros(num_test,1);
nmse = zeros(num_test,1);
psnrer = zeros(num_test,1);

Xtest = Unfold(Xtest,size(Xtest),N+1);
for j = 1:num_test
    fprintf('Testing Sample %d: ',j)
    test = Xtest(j,:);
    test = Fold(test,dim,N+1);
    % Estimate the core tensor of the samples using the learned basis matrices
    Grec{j} = Estimate_core(test,PDD,patch_size,RR);
    % Quantize the core tensor using the learned symbols
    ll = Grec{j}(:);
    GG_quant{j} = zeros(length(ll),1);
    GG_enc = zeros(length(ll),1);
    for i = 1:length(ll)
        l = find(symbols<ll(i));
        u = find(symbols>ll(i));
        if isempty(u)
            GG_quant{j}(i) = length(symbols); 
        elseif isempty(l)
            GG_quant{j}(i) = 1; 
        else
            if ll(i)<=(symbols(l(end))+(symbols(u(1))-symbols(l(end)))/2)
                GG_quant{j}(i) = l(end);
            else
                GG_quant{j}(i) = u(1);
            end
        end
        GG_enc(i) = symbols(GG_quant{j}(i));
    end
    GG_quant{j} = reshape(GG_quant{j},size(Grec{j}));      
    % Huffman coding
    hcode = huffmanenco(GG_enc,dict); % encoded measurements - the data we send
    bits_coding(j) = bits_coding(j)+numel(hcode); % number of bits of the encoded sample
    GG_dec{j} = huffmandeco(hcode,dict); % decoded measurements
    GG_dec{j} = reshape(GG_dec{j},size(Grec{j})); 
    % Reconstruct the sample
    Mrec{j} = reconstruction(GG_dec{j},PDnew,patch_size,RR,dim);
    % Compute the error
    bpppb(j) = bits_coding(j)/prod(size(test)); % bits per pixel per band used
    nmse(j) = norm(Mrec{j}(:)-test(:))/norm(test(:));
    rmse = sqrt(sum((test(:)-Mrec{j}(:)).^2)/prod(size(test)));
    psnrer(j) = 20*log10(max(test(:)-min(test(:)))/(2*rmse));
    fprintf('NRMSE %4f, PSNR %2f dB, bpppb %2f\n',nmse(j),psnrer(j),bpppb(j))
end

fprintf('Mean Testing NRMSE %4f, PSNR %2f, bpppb %2f\n',mean(nmse),mean(psnrer),mean(bpppb))

%% Save the results
save('LFADMMP_3D_Chania_R05_50train_50test_8patch_lloyd8bit_huffman.mat','PDnew','symbols','bpppb','bits_coding','nmse','psnrer','Mrec','Grec','GG_quant')
