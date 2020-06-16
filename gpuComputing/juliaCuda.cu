// Vector version
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
