/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

void __global__ shmem_reduce_max_kernel(float* d_out,const float* const d_in,
    const size_t numRows, const size_t numCols)
{
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float sdata[];

  int myId_x = threadIdx.x + blockDim.x * blockIdx.x;
  int myId_y = threadIdx.y + blockDim.y * blockIdx.y;

  if(myId_x<numCols && myId_y<numRows)
  {
    // Find the Id of the thread in the global memory
    int myId = myId_y*numCols + myId_x;

    // Find the Id of the thread in the block
    int tid = threadIdx.y*blockDim.x + threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // Do reduction in shared mem
    // Divide the set into 2 and do the operations between them. Again split by
    // half and compare. Repeat till you get the single reduced element.
    for (unsigned int s = (blockDim.x*blockDim.y)/ 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
          sdata[tid]  = (sdata[tid]>sdata[tid + s]) ? sdata[tid]:sdata[tid+s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
      d_out[blockIdx.y*gridDim.x + blockIdx.x] = sdata[0];
    }
  }
}

void __global__ shmem_reduce_min_kernel(float* d_out,const float* const d_in,
    const size_t numRows, const size_t numCols)
{
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float sdata[];

  int myId_x = threadIdx.x + blockDim.x * blockIdx.x;
  int myId_y = threadIdx.y + blockDim.y * blockIdx.y;

  if(myId_x<numCols && myId_y<numRows)
  {
    // Find the Id of the thread in the global memory
    int myId = myId_y*numCols + myId_x;

    // Find the Id of the thread in the block
    int tid = threadIdx.y*blockDim.x + threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // Do reduction in shared mem
    // Divide the set into 2 and do the operations between them. Again split by
    // half and compare. Repeat till you get the single reduced element.
    for (unsigned int s = (blockDim.x*blockDim.y)/ 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid]  = (sdata[tid]<sdata[tid + s]) ? sdata[tid]:sdata[tid+s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
      d_out[blockIdx.y*gridDim.x + blockIdx.x] = sdata[0];
    }
  }
}

void reduce(const float* const d_in, float &reduced_loglum, std::string ops,
  const dim3 blockSize, const dim3 gridSize, const size_t numRows,
  const size_t numCols)
{
  int gridBytes = gridSize.x*gridSize.y*sizeof(float);
  int blockBytes = blockSize.x*blockSize.y*sizeof(float);

  float *d_intermediate, *d_out;

  checkCudaErrors(cudaMalloc((void**) &d_intermediate, gridBytes));
  checkCudaErrors(cudaMalloc((void**) &d_out, sizeof(float)));

  if(ops=="min")
  {
    shmem_reduce_min_kernel<<<gridSize, blockSize, blockBytes>>>(d_intermediate, d_in,
                                                                  numRows, numCols);
  }
  else if(ops=="max")
  {
    shmem_reduce_max_kernel<<<gridSize, blockSize, blockBytes>>>(d_intermediate, d_in,
                                                                  numRows, numCols);
  }
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // now we're down to one block left, so reduce it
  if(ops=="min")
  {
    shmem_reduce_min_kernel<<<1, gridSize, gridBytes>>>(d_out, d_intermediate,
                                                        numRows, numCols);
  }
  else if(ops=="max")
  {
    shmem_reduce_max_kernel<<<1, gridSize, gridBytes>>>(d_out, d_intermediate,
                                                        numRows, numCols);
  }
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&reduced_loglum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_intermediate));
  checkCudaErrors(cudaFree(d_out));
}


void __global__ generateHistogram(unsigned int* d_out, const float* const d_in,
  const float min, const float range, const size_t numRows,
  const size_t numCols, const size_t numBins)
{
  int myId_x = threadIdx.x + blockDim.x * blockIdx.x;
  int myId_y = threadIdx.y + blockDim.y * blockIdx.y;

  if(myId_x<numCols && myId_y<numRows)
  {
    // Find the Id of the thread in the global memory
    int myId = myId_y*numCols + myId_x;
    int bin = (int)(((d_in[myId] - min) / range) * numBins);
    atomicAdd(&(d_out[bin]), 1);
  }

}

void __global__ cdfScanHillisSteele(const unsigned int* const d_histogram,
  unsigned int* const d_cdf)
{
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float sdata[];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    // For 1 we need to implement the parallel reduce with min and max ops. They
    // are binary as well as associate operations and can be implemented in reduce.

    // For 2 just need to subtract the values

    // For 3 need to make a histogram using one of the methods. Lets start off
    // with the purely atomics one.

    // For 4 we need to implement an exclusive scan  for which we could use the
    // Reduce downsweep or modify hillis/steele for inclusive scan by adding the
    // identity to the list and performing the same steps.


// ==================== 1. Min Max using parallel reduce =======================

  // Set reasonable block size (i.e., number of threads per block)
  int blockWidth = 32;
  const dim3 blockSize(blockWidth, blockWidth, 1);

  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  int   blocksX = numCols/blockWidth+1;
  int   blocksY = numRows/blockWidth+1;
  const dim3 gridSize(blocksX,blocksY,1);

  // Call the reduce method
  reduce(d_logLuminance, min_logLum, "min", blockSize, gridSize, numRows, numCols);
  reduce(d_logLuminance, max_logLum, "max", blockSize, gridSize, numRows, numCols);

// ==================== 2. Find the range ======================================
  float range_logLum = max_logLum - min_logLum;
  std::cout<<"Min: "<<min_logLum<<" , "<<"Max: "<<max_logLum<<" , "<<"Range: "<<range_logLum<<std::endl;

// ==================== 3. Generate the histogram ==============================
  unsigned int* d_histogram;
  int binBytes = numBins*sizeof(unsigned int);
  checkCudaErrors(cudaMalloc((void**) &d_histogram, binBytes));
  checkCudaErrors(cudaMemset((void *) d_histogram, 0, binBytes));

  generateHistogram<<<gridSize, blockSize>>> (d_histogram, d_logLuminance, min_logLum,
    range_logLum, numRows, numCols, numBins);
  std::cout<<"Rows: "<<numRows<<" , "<<"Cols: "<<numCols<<" , "<<"Bins: "<<numBins<<std::endl;

  unsigned int h_histogram[numBins];
  for(unsigned int i = 0; i < numBins; i++) {
      h_histogram[i] = 0;
  }
  checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, binBytes, cudaMemcpyDeviceToHost));

  unsigned int sum = 0;
  for(unsigned int i=0; i<numBins;i++)
  {
    sum+=h_histogram[i];
  }

  std::cout<<"Sum: "<<sum<<" , "<<"Num pixels: "<<(numRows*numCols)<<std::endl;
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

// ==================== 4. Implement the HIllis/Steele Scan for cdf=============
  cdfScanHillisSteele<<<1, numBins, binBytes>>> (d_histogram, d_cdf);
  checkCudaErrors(cudaFree(d_histogram));
}
