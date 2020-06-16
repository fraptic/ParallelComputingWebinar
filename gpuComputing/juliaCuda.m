function [tcomp, Z] = juliaCuda(vis)
% JULIACUDA Computation and Visualization of the Julia set 
% for f(z) = z^2 - 0.8 + 0.156i  
% Calculation is done on GPU using MATLAB gpu functions
% http://en.wikipedia.org/wiki/Julia_set
%
% TCOMP = JULIACUDA calculates the Julia set. 
%
% TCOMP = JULIACUDA(vis) plots the result if vis = true 
%
% [TCOMP,Z] = JULIACUDA(...) returns the calculated Julia set
%
% See also: JULIACUDA.CU

%% Check if GPU is supported
checkGPU

%% Check input parameter
% default: only calculation, no plot
if nargin < 1
    vis = false;
end

%% Create the data
c = -0.8 + 0.156i;

tic
x = linspace(gpuArray(-1.5),1.5,4000);
y = linspace(gpuArray(-1),1,2000);
[X,Y] = meshgrid(x,y);
N = numel(X);

%% Generation of CUDA kernel
k = parallel.gpu.CUDAKernel('juliaCuda.ptx', 'juliaCuda.cu');
% product of Gridsize and ThreadBlockSize needs to be equal to number of
% elements in vector

mygpu = gpuDevice;
[blockSize, numThreads] = largestDivisor(2000*4000, ...
    mygpu.MaxThreadsPerBlock);
k.ThreadBlockSize = blockSize;
k.GridSize = numThreads;

%% Execution of CUDA kernel
Z = gather(feval(k,X,Y,real(c),imag(c),N));
tcomp = toc;

disp(['Elapsed time using CUDA kernels: ' num2str(tcomp) ' seconds.'])

%% Visualization of the Julia set
% Note: if the data is just generated for visualization, this
% can also be directly applied to gpuArrays
if vis
    visJulia(x,y,Z)
end
