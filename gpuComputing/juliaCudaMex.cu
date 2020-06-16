/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"

/*
 * Device code
 */
__global__ void Iteration(double *Xreal, double *Ximag, 
        const double creal, const double cimag, const double N) 
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int i = threadIdx.x; // if using just one block
    int k;
    double temp;
    if (i < N) {
        for (k = 0; k < 100; k++){
            temp = 2 * Xreal[i] * Ximag[i] + cimag;
            Xreal[i] = Xreal[i] * Xreal[i] - Ximag[i] * Ximag[i] + creal;
            Ximag[i] = temp;
        }
        // Xreal is the only output that needs to be retrieved
        Xreal[i] = exp(-sqrt(Xreal[i] * Xreal[i] + Ximag[i] * Ximag[i]));
    }
}

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const *X_real;
    mxGPUArray const *X_imag;
    double c_real;
    double c_imag;
    double N;

    double *d_real;
    double *d_imag;
    int dim;

    char const * const errId = "julia_CUDA_mex:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 512;
    int blocksPerGrid;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if (nrhs!=5) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    X_real = mxGPUCreateFromMxArray(prhs[0]);
    X_imag = mxGPUCreateFromMxArray(prhs[1]);
    c_real = mxGetScalar(prhs[2]);
    c_imag = mxGetScalar(prhs[3]);
    N = mxGetScalar(prhs[4]);

    /*
     * Verify that inputs really are  double arrays before extracting the pointer.
     */
    if ((mxGPUGetClassID(X_real) != mxDOUBLE_CLASS) || 
       (mxGPUGetClassID(X_imag) != mxDOUBLE_CLASS)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_real = (double *)(mxGPUGetDataReadOnly(X_real));
    d_imag = (double *)(mxGPUGetDataReadOnly(X_imag));

    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
    dim = (int)(mxGPUGetNumberOfElements(X_real));
    blocksPerGrid = (dim + threadsPerBlock - 1) / threadsPerBlock;
    Iteration<<<blocksPerGrid, threadsPerBlock>>>(d_real, d_imag, c_real, c_imag, N);

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(X_real);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(X_real);
    mxGPUDestroyGPUArray(X_imag);
}
