#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void cube(float * d_out, float * d_in){
	// Todo: Fill in this function
	int id = threadIdx.x;
	float num  = d_in[id];
	d_out[id] = num*num*num;
}

int main(int argc, char ** argv) {

	// NOTE: h is for host and d is for device
	// This is the general template of cuda code
	// 1. CPU allocate memory in the device.
	// 2. CPU copy data from the host structs to the device structs
	// 3. CPU runs the kernel on the GPU
	// 4. CPU copies the data from the device struct to the host struct

	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE];

	// declare GPU memory pointers
	float * d_in;
	float * d_out;

	// allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	// The first is the number of blocks and the second is the number of threads per block
	// Can run many blocks at once.
	// Max number of threads per block is 512 for old and 1024 for new
	cube<<<1, ARRAY_SIZE>>>(d_out, d_in);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
