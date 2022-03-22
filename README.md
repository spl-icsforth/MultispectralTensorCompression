# MultispectralTensorCompression
Tensor Decomposition Learning for Compression of Multidimensional Signals

This repository contains MATLAB codes and scrips designed for the compression of multidimensional signals 
based on a novel tensor decomposition learning method that uses the Tucker decomposition, as it is presented 
in the paper "Tensor Decomposition Learning for Compression of Multidimensional Signals" (A. Aidini, G. Tsagkatakis, 
P. Tsakalides). In the proposed method, a generic basis for each dimension is learned from a set of training samples
that can efficiently represent every sample, by solving a constrained optimization problem using the Alternating 
Direction Method of Multipliers (ADMM). Given the learned basis matrices, each new sample is firstly represented
as a multilinear combination of them with coefficients in a reduced-sized core tensor. Then, the derived coefficients
are quantized and encoded in order to be transmitted, using a symbol encoding dictionary that is also learned in the
training process.


**Requirements**

[Dataset](https://drive.google.com/drive/folders/1IesESYVA7xNJcvWnKk3l2DrTvx54EcYQ?usp=sharing)

The efficacy of the proposed compression algorithm is evaluated over publicly available remote sensing multispectral
images daily acquired by the MODIS satellite over the region of Chania in Crete. In more detail, the files 
NASA2017_Chania.mat and NASA2018_Chania.mat contain 365 images of 2017 that are used for the training set and the 
corresponding 365 images of 2018 for the testing set, respectively, each of them modeled as a third-order tensor of size
227 x 348 x 5 with two spatial and one spectral dimension.

Tensor Toolbox

We use the tensor toolbox for MATLAB, which is available in https://www.tensortoolbox.org and contains useful 
functions for tensor operators.

Lloyd-max quantization

We use Lloyd-max quantization, which is implemented in https://github.com/papanikge/pcm-matlab/blob/master/lloyd_max.m
(files: lloyd_max.m and my_quantizer.m)


**Contents**

main.m : The primary script that loads the data, performs the compression using the proposed tensor decomposition
learning method in combination with quantization and encoding of the tensor data, and provides the results.

Tensor_Decomposition_Learning.m: Perform the tensor decomposition learning method during the training process.

Estimate_core.m : Estimate the core tensor of the samples using the learned basis matrices.

reconstruction.m : Reconstruct the samples from the received data using the learned basis matrices.

patches.m : Take the patches of a sample across the spatial dimensions.

union_patches.m : Stitch the patches to create the sample.

Unfold.m : Unfold the tensor into a matricization mode.

Fold.m : Fold a matricization mode into a tensor.


**References**

1. A.  Aidini,  G.  Tsagkatakis,  and P.  Tsakalides, [“Tensor Decomposition Learning for Compression of
Multidimensional Signals,”](http://users.ics.forth.gr/~tsakalid/PAPERS/2021-JSTSP.pdf) IEEE Journal of Selected Topics in Signal Processing 15.3 (2021): 476-490

2. Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox Version 3.1, Available online, June 2019.
URL: https://gitlab.com/tensors/tensor_toolbox.
