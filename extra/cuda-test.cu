#include <stdio.h>

class CudaClass
{
public:
int* data;
CudaClass(int x) {
    data = new int[1]; data[0] = x;
}
};

__global__ void useClass(CudaClass *cudaClass)
{
    printf("%d\n", cudaClass->data[0]);
};




int main()
{
    CudaClass c(1);
    // create class storage on device and copy top level class
    CudaClass *d_c;
    cudaMalloc((void **)&d_c, sizeof(CudaClass));
    cudaMemcpy(d_c, &c, sizeof(CudaClass), cudaMemcpyHostToDevice);
    // make an allocated region on device for use by pointer in class
    int *hostdata;
    cudaMalloc((void **)&hostdata, sizeof(int));
    cudaMemcpy(hostdata, c.data, sizeof(int), cudaMemcpyHostToDevice);
    // copy pointer to allocated device storage to device class
    cudaMemcpy(&(d_c->data), &hostdata, sizeof(int *), cudaMemcpyHostToDevice);
    useClass<<<1,1>>>(d_c);
    cudaDeviceSynchronize();
    return 0;
}
