//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include<bitset>
#define RADIX_NUMBER 2 // This is the log of number of radixbits
/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   =============================
   =============

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.
 */

  // Aggregate the output data using all the information calculated
  void __global__ generateOutput(unsigned int* const d_outputVals,
    unsigned int* const d_outputPos, const unsigned int* const d_inputVals,
     const unsigned int* const d_inputPos,
     const unsigned int* const d_relative_offsets,
     const unsigned int* const d_local_cdf, unsigned int* const d_cdf,
     const unsigned int* const d_predicate, const size_t numBins,
     const size_t numElems)
   {
     int tid = threadIdx.x;
     int bid = blockIdx.x;
     int myId = tid + blockDim.x * bid;

     if(myId<numElems)
     {
       int predicate = d_predicate[myId];

       int in_grid_position = d_relative_offsets[myId];
       int in_bin_position = d_local_cdf[bid*numBins + predicate] + in_grid_position;
       int global_position = d_cdf[predicate] + in_bin_position;

       if(global_position<numElems)
       {
         d_outputVals[global_position] = d_inputVals[myId];
         d_outputPos[global_position] = d_inputPos[myId];
       }
     }
   }

  // Generate the relative offsets of each set using a compact and segmented scan
  // Note that this is within a single thread block
  void __global__ generateScanRelativeOffsets(unsigned int* const d_relative_offsets,
    const unsigned int* const d_compact, const size_t numElems,
    const size_t numBins)
  {
    // soffset data is numBins x blockDim.x
    extern __shared__ unsigned int sscandata[];

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int grid = bid/numBins;
    unsigned int bin = bid%numBins;
    unsigned int linearId = tid + grid*blockDim.x;

    unsigned int n = blockDim.x;
    unsigned int pout = 0;
    unsigned int pin = 1;


    unsigned int compact_val = 0;

    if(linearId<numElems)
      compact_val = d_compact[numElems*bin + linearId];

    sscandata[pout*n+tid] = compact_val;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x ; s <<= 1)
    {
      // Swap pin and pout
      pout = 1 - pout;
      pin = 1 - pout;
      if(linearId<numElems)
      {
        // Do not touch if the data has no neighbour s indexes left to it.
        if(tid>=s)
        {
          sscandata[pout*n+tid] = sscandata[pin*n+tid] + sscandata[pin*n+tid-s];
        }
        else
        {
          sscandata[pout*n+tid] = sscandata[pin*n+tid];
        }
      }
      __syncthreads();        // make sure all adds at one stage are done!
    }
    if(linearId<numElems && compact_val==1)
    {
      // You need to subtract by -1 to ensure exclusive scan
      d_relative_offsets[linearId] = sscandata[pout*n+tid]-1;
    }
  }

  // Generate the compact using the predicate
  void __global__ generateCompact(unsigned int* const d_compact,
    const unsigned int* const d_predicate, const unsigned int numBins,
    const unsigned int numElems)
  {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    // unsigned int myId = tid + bid*blockDim.x;
    unsigned int grid = bid/numBins;
    unsigned int bin = bid%numBins;

    unsigned int linearId = tid + grid*blockDim.x;

    if(linearId<numElems)
      d_compact[numElems*bin+linearId] = (d_predicate[linearId]==bin) ? 1:0;
  }

  // Generate the global cdf
  void __global__ generateGlobalCdf(unsigned int* const d_cdf,
    const unsigned int* const d_histogram)
  {
    extern __shared__ float sglobalhisto[]; // allocated on invocation

    int n = blockDim.x;
    int bin = threadIdx.x;
    int pout = 0, pin = 1;

    // This is exclusive scan, so shift right by one and set first element to 0
    sglobalhisto[pout*n + bin] = (bin > 0) ? d_histogram[bin-1] : 0;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2)
    {
      pout = 1 - pout; // swap double buffer indices
      pin = 1 - pout;
      if (bin >= s)
        sglobalhisto[pout*n+bin] = sglobalhisto[pin*n+bin] + sglobalhisto[pin*n+bin-s];
      else
        sglobalhisto[pout*n+bin] = sglobalhisto[pin*n+bin];
     __syncthreads();
    }
    d_cdf[bin] = sglobalhisto[pout*n+bin]; // write output
  }

  // Generate the global histogram
  void __global__ generateGlobalHistogram( unsigned int* const d_histogram,
    const unsigned int* const d_local_cdf, const unsigned int* const d_local_histogram,
    const size_t gridSize)
  {
    int bin = threadIdx.x;
    int idx_final_bin = blockDim.x*(gridSize-1) + bin;
    d_histogram[bin] = d_local_cdf[idx_final_bin] + d_local_histogram[idx_final_bin];
  }

  // Generate the local cdf of the histograms along the grid. Here the d_histogram is of size
  // gridSize x numBins
  void __global__ generateLocalCdf( unsigned int* const d_local_cdf,
    const unsigned int* const d_local_histogram, const size_t numBins)
  {
    extern __shared__ float shistodata[]; // allocated on invocation

    // Thread id
    int grid = threadIdx.x;
    int bin = blockIdx.x;

    // Tells me which part is storing the in and out at the current step
    int pout = 0, pin = 1;

    // Store the total number of elements
    int n = blockDim.x;

    // Copy the data to the shared memory
    // shistodata[tid] = d_local_histogram[tid];
    // If we're doing an exclusive scan copy the data from n-1 and set position
    // 0 to identity which in this case is 0
    shistodata[pout*n + grid] = (grid>0) ? d_local_histogram[(grid-1)*numBins + bin] : 0;
    __syncthreads();

    // Do the hillis steele reduction along the grid dimension only
    for (int s = 1; s < blockDim.x; s *= 2)
    {
      pout = 1 - pout; // swap double buffer indices
      pin = 1 - pout;

      if (grid >= s)
          shistodata[pout*n + grid] = shistodata[pin*n + grid] + shistodata[pin*n + grid-s];
      else
          shistodata[pout*n + grid] = shistodata[pin*n + grid];
      __syncthreads();
    }

    // Populate the grid row of the cdf
    d_local_cdf[grid*numBins + bin] = shistodata[pout*n + grid];
  }

  // Generate local histograms and populate the predicate
  void __global__ generateLocalHistograms(unsigned int* const d_local_histogram,
    unsigned int* const d_predicate, const unsigned int* const d_inputVals,
    const size_t numElems, const size_t numBins, const unsigned int mask,
    const unsigned int shift)
  {
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int bid = blockIdx.x;

    if(myId<numElems)
    {
      unsigned int val = d_inputVals[myId];

      // Find the digit by applying the mask
      unsigned int digit = (val&mask)>>shift;

      // Update the predicate
      d_predicate[myId] = digit;

      // Add the point to the local histogram at position (grid,digit)
      atomicAdd(&(d_local_histogram[bid*numBins + digit]), 1);
    }
  }

  // The main radix step
  void radixSortStep(const unsigned int* const d_inputVals,
    const unsigned int* const d_inputPos, unsigned int* const d_outputVals,
    unsigned int* const d_outputPos, unsigned int* const d_local_histogram,
    unsigned int* const d_local_cdf, unsigned int* const d_histogram,
    unsigned int* const d_cdf, unsigned int* const d_predicate,
    unsigned int* const d_compact, unsigned int* const d_relative_offsets,
    const size_t blockSize, const size_t gridSize, const size_t numBins,
    const size_t numElems, const unsigned int radixBits, const int place,
    const unsigned int seed_mask)
  {
//////////////Define the shift while evaluating this place//////////////////////
    unsigned int shift = place*radixBits;
    unsigned int mask = seed_mask<<shift;

    // NOTE: Debug
    // std::cout<<"Mask "<<place<<" : "<<std::bitset<sizeof(unsigned int)*8>(mask)<<std::endl;
    // std::cout<<"numBins "<<place<<" : "<<numBins<<std::endl;
    // std::cout<<"numElems "<<place<<" : "<<numElems<<std::endl;

//////Populate the local histogram (gridSize x numBins) and the predicates//////
    generateLocalHistograms<<<gridSize, blockSize>>>(d_local_histogram,
      d_predicate, d_inputVals, numElems, numBins, mask, shift);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // NOTE: Debug
    // unsigned int h_local_histogram[gridSize*numBins];
    // unsigned int sum=0;
    // checkCudaErrors(cudaMemcpy(h_local_histogram, d_local_histogram, gridSize*numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // for(size_t i=0;i<gridSize*numBins;i++)
    // {
    //   if(h_local_histogram[i]>1024)
    //     std::cout<<h_local_histogram[i]<<" ";
    //   sum+=h_local_histogram[i];
    // }
    // std::cout<<sum<<std::endl;

///// Populate the local cdf (gridSize x numBins)along the grid dimension //////
    // It is this local cdf that enables connecting the scans in individual
    // thread blocks
    generateLocalCdf<<<numBins, gridSize, 2*gridSize*sizeof(unsigned int)>>>(
      d_local_cdf, d_local_histogram, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // NOTE: Debug
    // unsigned int h_local_cdf[gridSize*numBins];
    // unsigned int sum=0;
    // checkCudaErrors(cudaMemcpy(h_local_cdf, d_local_cdf, gridSize*numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // for(size_t i=(gridSize-1)*numBins;i<gridSize*numBins;i++)
    // {
    //     sum+=(h_local_cdf[i]+h_local_histogram[i]);
    // }
    // std::cout<<sum<<std::endl;

///////////////// Populate the global histogram (numBins) //////////////////////
    generateGlobalHistogram<<<1, numBins>>>(d_histogram, d_local_cdf, d_local_histogram,
      gridSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //NOTE: Debug
    // unsigned int h_histogram[numBins];
    // int sum = 0;
    // checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // for(size_t i=0;i<numBins;i++)
    // {
    //   // std::cout<<h_histogram[i]<<" ";
    //   sum+=h_histogram[i];
    // }
    // std::cout<<sum<<std::endl;

//////////////// Populate the global cdf (gridSize)/////////////////////////////
    generateGlobalCdf<<<1, numBins, 2*numBins*sizeof(unsigned int)>>>(d_cdf,
      d_histogram);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //NOTE: Debug
    // unsigned int h_cdf[numBins];
    // checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // for(size_t i=0;i<numBins;i++)
    // {
    //   std::cout<<h_cdf[i]<<" ";
    // }
    // std::cout<<std::endl;
    // std::cout<<h_cdf[numBins-1]+h_histogram[numBins-1]<<std::endl;

////////////////// Generate the compact (numBins x numElems) list///////////////
    // Note that the size of d_compact is numBins x numElems where each row
    // represents the compact of each element for a particular bin/digit
    generateCompact<<<gridSize*numBins, blockSize>>>(d_compact, d_predicate,
      numBins, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // NOTE: Debug
    // unsigned int* h_compact = new unsigned int[numBins*numElems];
    // checkCudaErrors(cudaMemcpy(h_compact, d_compact, numBins*numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // unsigned int sum=0;
    // for(size_t i=0;i<numBins*numElems;i++)
    // {
    //   sum+=h_compact[i];
    // }
    // std::cout<<sum<<std::endl;
    // delete h_compact;

/////////////// Estimate the relative offsets (numElems) in each block /////////
    //Note that the relative offsets are the offsets of the digit of the element
    // in that "place" in the block for that digit. You need to get the offset
    // between blocks for each digit using the local cdf. You need to get the
    // global offset between digits using the global cdf.
    // So:
    // relative_offset:  Facilitates within block and digit position estimation
    // local_cdf: Facilitates within digit position estimation
    // global_cdf: Facilitates total position estimation
    generateScanRelativeOffsets<<<gridSize*numBins, blockSize,
      2*blockSize*sizeof(unsigned int)>>>(d_relative_offsets, d_compact, numElems,
        numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


///////////// Estimate the final position of the block and move the data///////
    // 1. Use the relative offset for offset within a block
    // 2. Use the local_cdf for grid based offset for offset within a bin
    // 3. Use the global cdf to compute the offset in the entire list
    // Use the predicate to access the correct bin in each of the above cases.
    generateOutput<<<gridSize, blockSize>>>(d_outputVals, d_outputPos, d_inputVals,
      d_inputPos, d_relative_offsets, d_local_cdf, d_cdf, d_predicate, numBins,
      numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

///////////////////////////////////END//////////////////////////////////////////
  }


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  //TODO
  //PUT YOUR SORT HERE
  // So a radix sort is you take the LSB move all the 0s to the top and all the 1s
  // to the bottom

  // Use powers of 2 as the radix bits
  // This is important so that the numPlaces are even
  const unsigned int radixBits = pow(2,RADIX_NUMBER);
  const unsigned int seed_mask = pow(2,radixBits)-1;
  const size_t numBins = pow(2,radixBits);

  // Assuming unsigned int size of 32 number of places equals 32/radixBits
  const size_t numPlaces = sizeof(unsigned int)*8/radixBits;
  int binBytes = numBins * sizeof(unsigned int);
  int elemBytes = numElems * sizeof(unsigned int);
  int allBytes = numBins * numElems * sizeof(unsigned int);

  // Assign the block and the grid size
  const size_t blockSize = 1024;
  const size_t gridSize = numElems/blockSize + 1;

  // Contains the local per grid histogram information
  // Has size equals gridSize x numBins. It contains the histogram per grid
  unsigned int* d_local_histogram;
  checkCudaErrors(cudaMalloc((void**) &d_local_histogram, gridSize*binBytes));
  checkCudaErrors(cudaMemset((void *) d_local_histogram, 0, gridSize*binBytes));

  // Contains the local per grid cdf information cdf along the grid dimension
  // Has size equals gridSize x numBins. It contains the cdf in the grid dimension
  unsigned int* d_local_cdf;
  checkCudaErrors(cudaMalloc((void**) &d_local_cdf, gridSize*binBytes));
  checkCudaErrors(cudaMemset((void *) d_local_cdf, 0, gridSize*binBytes));

  // Contains the global histogram information
  // Has size equals numBins representing all possible digits in a place
  unsigned int* d_histogram;
  checkCudaErrors(cudaMalloc((void**) &d_histogram, binBytes));
  checkCudaErrors(cudaMemset((void *) d_histogram, 0, binBytes));

  // Contains the global cdf per bin
  // Has size equals numBins representing all possible digits in a place
  unsigned int* d_cdf;
  checkCudaErrors(cudaMalloc((void**) &d_cdf, binBytes));
  checkCudaErrors(cudaMemset((void *) d_cdf, 0, binBytes));

  // This contains the digit at a place information for each element
  // Has size equals numElems
  unsigned int* d_predicate;
  checkCudaErrors(cudaMalloc((void**) &d_predicate, elemBytes));
  checkCudaErrors(cudaMemset((void *) d_predicate, 0, elemBytes));

  // This contains the relative offsets of each element
  // Has size equals numElems
  unsigned int* d_relative_offsets;
  checkCudaErrors(cudaMalloc((void**) &d_relative_offsets, elemBytes));
  checkCudaErrors(cudaMemset((void *) d_relative_offsets, 0, elemBytes));

  // Make an array to store the compact list for each bin
  // Has size equal numBins x numElems
  unsigned int* d_compact;
  checkCudaErrors(cudaMalloc((void**) &d_compact, allBytes));
  checkCudaErrors(cudaMemset((void *) d_compact, 0, allBytes));

  // I'm guaranteed to have even number of places with the way the RADIX NUMBER
  // and radix digits are defined. So I don't need to check whether the output
  // arrays actually contain the final output.
  for(unsigned int i=0; i<numPlaces; i++)
  {
    // Do a step of the radix Sort, swap input and output each step
    if(i%2==0)
      radixSortStep(d_inputVals, d_inputPos, d_outputVals, d_outputPos,
        d_local_histogram, d_local_cdf, d_histogram, d_cdf, d_predicate,
        d_compact, d_relative_offsets, blockSize, gridSize, numBins, numElems, radixBits,
        i, seed_mask);
    else
      radixSortStep(d_outputVals, d_outputPos, d_inputVals, d_inputPos,
        d_local_histogram, d_local_cdf, d_histogram, d_cdf, d_predicate,
        d_compact, d_relative_offsets, blockSize, gridSize, numBins, numElems,
        radixBits, i, seed_mask);

    // Reset all the data to 0 for the next step
    checkCudaErrors(cudaMemset((void *) d_local_histogram, 0, gridSize*binBytes));
    checkCudaErrors(cudaMemset((void *) d_local_cdf, 0, gridSize*binBytes));
    checkCudaErrors(cudaMemset((void *) d_histogram, 0, binBytes));
    checkCudaErrors(cudaMemset((void *) d_cdf, 0, binBytes));
    checkCudaErrors(cudaMemset((void *) d_predicate, 0, elemBytes));
    checkCudaErrors(cudaMemset((void *) d_relative_offsets, 0, elemBytes));
    checkCudaErrors(cudaMemset((void *) d_compact, 0, allBytes));
  }

  // Note that the numPlaces are even so the final ouput should be in the output
  // fields

  // Free all the assigned arrays
  checkCudaErrors(cudaFree(d_compact));
  checkCudaErrors(cudaFree(d_relative_offsets));
  checkCudaErrors(cudaFree(d_predicate));
  checkCudaErrors(cudaFree(d_cdf));
  checkCudaErrors(cudaFree(d_histogram));
  checkCudaErrors(cudaFree(d_local_cdf));
  checkCudaErrors(cudaFree(d_local_histogram));

}
