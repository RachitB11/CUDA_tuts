//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

#include<bitset>

#define RADIX_NUMBER 2
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

 // Use the predicate and the histogram to map the data to the new positions in
 // the sorted array
 void __global__ fillOutputValAndPositions(const unsigned int* const d_input,
   const unsigned int* const d_inputPos,
   unsigned int* const d_output, unsigned int* const d_outputPos,
   const unsigned int* const d_histogram, const unsigned int* const d_relative_offsets,
   const unsigned int* const d_predicate, const size_t numElems)
 {
   int myId = threadIdx.x + blockDim.x * blockIdx.x;

   if(myId<numElems)
   {
     unsigned int new_position = d_histogram[d_predicate[myId]] + d_relative_offsets[myId];
     d_outputPos[new_position] = d_inputPos[myId];
     d_output[new_position] = d_input[myId];
   }
 }

 // For a thread make a 16xnumel shared array in which each of the dimensions is initialized to 1 at places where
 void __global__ compactAndSegmentScan( const unsigned int* const d_inputPos,
   const unsigned int* const d_predicate, unsigned int* const d_relative_offsets,
   const size_t numBins, const size_t numElems)
 {
   // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
   // Another quick note here we are doing dynamic memory allocation.
   extern __shared__ unsigned int soffsetdata[];

   printf("Here1");

   int myId = threadIdx.x + blockDim.x * blockIdx.x;

   soffsetdata[myId]=1;
   __syncthreads();

   printf("Here2");

   unsigned int temp;

   for (unsigned int s = 1; s < numElems ; s <<= 1)
   {
       if(myId<numElems)
       {
         // Do not touch if the data has no neighbour s indexes left to it.
         if(myId >= s)
         {
            if(d_predicate[myId]==d_predicate[myId-s])
              temp = soffsetdata[myId] + soffsetdata[myId - s];
         }
         else
         {
            temp = soffsetdata[myId];
         }
       }
       __syncthreads();        // make sure all adds at one stage are done!
       if(myId<numElems)
        soffsetdata[myId] = temp;
       __syncthreads();        // make sure all adds at one stage are done!
   }

   printf("Here3");

   if(myId<numElems)
   {
     // You need to subtract by -1 to ensure exclusive scan
     d_relative_offsets[myId] = soffsetdata[myId] - 1;
   }

   printf("Here4");
 }


 // Perform the hillis steele scan with the sum operation to get the cdf for each histogram
 // Here we perform a segmented scan since we're handling multiple histograms together.
 //NOTE: This assumes the gridSize is 1 and that the blockSize is a power of 2.
 // TODO: Try it by assigning double the memory and swapping the memory between
 // half that array between iterations to avoid having to add 2 __syncThreads in
 // the for loop.
 // https://stackoverflow.com/questions/19736560/hillis-steele-kernel-function
 void __global__ cdfSegmentedScanHillisSteele(const unsigned int* const d_histograms,
   unsigned int* const d_cdfs, const size_t numHistograms)
 {
   // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
   // Another quick note here we are doing dynamic memory allocation.
   extern __shared__ unsigned int shistodata[];

   // Find the Id of the thread in the block
   int tid = threadIdx.x;

   // Copy the data to the shared memory
   // shistodata[tid] = d_histogram[tid];
   // If we're doing an exclusive scan copy the data from n-1 and set position
   // 0 to identity which in this case is 0
   if(tid==0)
    for(unsigned int i=0; i<numHistograms; i++)
      shistodata[i*blockDim.x + tid] = 0;
   else
    for(unsigned int i=0; i<numHistograms; i++)
      shistodata[i*blockDim.x + tid] = d_histograms[i*blockDim.x + tid - 1];
   __syncthreads();

   unsigned int* temp = new unsigned int[numHistograms];

   for (unsigned int s = 1; s < blockDim.x ; s <<= 1)
   {
       // Do not touch if the data has no neighbour s indexes left to it.
       if(tid >= s)
       {
         for(unsigned int i=0; i<numHistograms; i++)
          temp[i] = shistodata[i*blockDim.x + tid] + shistodata[i*blockDim.x + tid - s];
       }
       else
       {
         for(unsigned int i=0; i<numHistograms; i++)
          temp[i] = shistodata[i*blockDim.x + tid];
       }
       __syncthreads();        // make sure all adds at one stage are done!
       for(unsigned int i=0; i<numHistograms; i++)
        shistodata[i*blockDim.x + tid] = temp[i];
       __syncthreads();        // make sure all adds at one stage are done!
   }

   // Assign to the global array
   for(unsigned int i=0; i<numHistograms; i++)
    d_cdfs[i*blockDim.x + tid] = shistodata[i*blockDim.x + tid];

   delete temp;
 }


 // Generate the histograms using the simple atomics way
 void __global__ generateHistograms(unsigned int* const d_histograms,
   unsigned int* const d_predicate, unsigned int* const d_inputVals,
   const unsigned int* const d_mask_array, const size_t numElems,
   const size_t numHistograms, const size_t numBins,
   const unsigned int radixBits)
 {
   int myId = threadIdx.x + blockDim.x * blockIdx.x;

   unsigned int val = d_inputVals[myId];

   if(myId<numElems)
   {
     for(unsigned int i=0; i<numHistograms; i++)
     {
       // Find the digit by applying the mask
       unsigned int digit = (val&d_mask_array[i])>>(i*radixBits);
       // Update the predicate
       d_predicate[i*numElems + myId] = digit;
       atomicAdd(&(d_histograms[i*numBins + digit]), 1);
     }
   }
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

  // TODO:
  // Convert the global histogram function into a local histogram function.
  // Do an inclusive scan on the local histogram data. The last elements are the global histogram data
  // The remaining in the inclusive scan will be used to estimate the relative offsets.
  // So this local histogram cdf can be used directly.

  // Local histogram (digit*grid*numBins) cdf along the grid dimension per digit 
  // will give the information in the offset while doing the scan over the whole
  // array for individual digits.
  // Global histogram (digit*numBins) cdf will give the offsets between digits.

////// 1. Histogram of the number of occurrences of each digit /////////////////

  // NOTE:
  // Digit is a block of the bitset. A "place" is of size radixBits.
  // So if radix bits is 4. There are 16 possible digits and 8 places (Assuming 32 bit integer)
  // Number of histograms equals number of places  = 8.
  // Number of bins is the number of possible digits = 16.

  // The digit is the number of symbols you account for in your radix sort. So for
  // a radix of 1 bit there will be 2 bins in the histogram. For 2 bits there
  // will be 4 bins, 3 bits there will be 8 bins and so on. So lets keep the num
  // bits as a varible.

  // Use powers of 2 as the radix bits
  // This is important so that the num histograms are a whole number
  const unsigned int radixBits = pow(2,RADIX_NUMBER);
  const size_t numBins = pow(2,radixBits);
  unsigned int mask = pow(2,radixBits)-1;

  // Assuming unsigned int size of 32 number of histograms equals number of masks
  // equals 32/radixBits
  const size_t numHistograms = 32/radixBits;
  int binBytes = numHistograms * numBins * sizeof(unsigned int);
  int elemBytes = numHistograms * numElems * sizeof(unsigned int);

  unsigned int* h_mask_array = new unsigned int[numHistograms];
  // Populate the mask array
  for(unsigned int i=0; i<numHistograms; i++)
  {
    h_mask_array[i] = mask<<(radixBits*i);
    // std::cout<<std::bitset<32>(h_mask_array[i])<<std::endl;
  }
  // Move the mask array to the device
  unsigned int* d_mask_array;
  checkCudaErrors(cudaMalloc((void**) &d_mask_array, numHistograms * sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_mask_array, h_mask_array, numHistograms * sizeof(unsigned int), cudaMemcpyHostToDevice));

  // Array to store all the histograms of digits
  // Digit j in histogram i can be accessed as (i*numBins + j)
  unsigned int* d_histograms;
  checkCudaErrors(cudaMalloc((void**) &d_histograms, binBytes));
  checkCudaErrors(cudaMemset((void *) d_histograms, 0, binBytes));

  // Find the predicate too while finding the histogram. That predicate will be
  // use to find the relative offset. The predicate here is just whether the
  // digit exists or not.
  // int j to represent whether Elem e has digit j in place i can be accessed as (i*numElems + e = j)
  unsigned int* d_predicate;
  checkCudaErrors(cudaMalloc((void**) &d_predicate, elemBytes));
  checkCudaErrors(cudaMemset((void *) d_predicate, 0, elemBytes));

  // Assign the block and the grid size
  const size_t blockSize = 1024;
  const size_t gridSize = numElems/blockSize + 1;

  // Generate numHistograms using the mask array
  // This should do 2 things:
  // - Populate the histogram (Size numHistograms x numBins) of digits for each place.
  // - Populate the predicate (Size numHistograms x numElems) to find digit in each place.
  generateHistograms<<<gridSize, blockSize>>> (d_histograms, d_predicate, d_inputVals,
    d_mask_array, numElems, numHistograms, numBins, radixBits);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

//////////////////2. Exclusive Prefix Sum of Histogram////////////////////unsigned //////
  // Allocate space for the cdf of the histograms.
  // Same size as d_histgrams (numHistograms x numBins)
  unsigned int* d_cdfs;
  checkCudaErrors(cudaMalloc((void**) &d_cdfs, binBytes));
  checkCudaErrors(cudaMemset((void *) d_cdfs, 0, binBytes));
  // Do an exclusive scan on each of the histograms
  cdfSegmentedScanHillisSteele<<<1, numBins, binBytes>>>(d_histograms, d_cdfs, numHistograms);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  std::cout<<"Number of elements are "<<numElems<<std::endl;
  unsigned int* h_histograms = new unsigned int[numHistograms*numBins];
  unsigned int* h_cdfs = new unsigned int[numHistograms*numBins];
  checkCudaErrors(cudaMemcpy(h_histograms, d_histograms, binBytes, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_cdfs, d_cdfs, binBytes, cudaMemcpyDeviceToHost));
  for(unsigned int i=0; i<numHistograms; i++)
  {
    unsigned int sum = 0;
    std::cout<<"CDF: ";
    for(unsigned int j=0; j<numBins; j++)
    {
      sum+=h_histograms[i*numBins+j];
      std::cout<<h_cdfs[i*numBins+j]<<" ";
    }
    std::cout<<std::endl<<"Sum for Histogram "<<i<<" the sum is "<<sum<<std::endl;
  }
  delete h_histograms;

  checkCudaErrors(cudaFree(d_mask_array));
  checkCudaErrors(cudaFree(d_histograms));
  delete h_mask_array;

//////////////////3.Determine relative offset of each digit/////////////////////
  // Basically go through the numbers 1 by 1 and check its digit and add offset by
  // 1
  // Do a relative offset on each digit at a particular place using the predicate array.

  // Allocate space for the relative offset of the histograms
  // Size (numBins x numElems)
  unsigned int* d_relative_offsets;
  checkCudaErrors(cudaMalloc((void**) &d_relative_offsets, elemBytes/numHistograms));
  checkCudaErrors(cudaMemset((void *) d_relative_offsets, 0, elemBytes/numHistograms));

  for(unsigned int i=0; i<numHistograms; i++)
  {
    // You need to ping pong between the output and input because they are on
    // const addresses and cannot be swapped.
    // You need to first create an auxillary block which will be of size numBinsxgridSize
    // which will store the segmented scan for each block.
    // While setting the final position, you will make use of 3 things to estimate
    // the final position:
    // - d_relative_offsets which contains the block offsets info (maintain a histogram locally)
    // - d_block_offsets which contain the offsets of each block (You need to do a scan over local histograms)
    // - d_cdf which gives you offset in each bin.
    // - Final should be (d_cdf + d_block_offsets + d_relative_offsets)


    // if(i%2==0)
    // {
    //   compactAndSegmentScan<<<gridSize,blockSize, blockSize*sizeof(unsigned int)>>>(
    //     d_inputPos, (d_predicate + (i*numElems)), d_relative_offsets, numBins,
    //     numElems);
    //   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    //
    //   fillOutputValAndPositions<<<gridSize,blockSize>>>(d_inputVals,
    //     d_inputPos, d_outputVals, d_outputPos,
    //     d_cdfs + (i*numBins), d_relative_offsets,
    //     d_predicate + (i*numElems), numElems);
    //   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // }
    // else
    // {
    //   compactAndSegmentScan<<<gridSize,blockSize, blockSize*sizeof(unsigned int)>>>(
    //     d_outputPos, (d_predicate + (i*numElems)), d_relative_offsets, numBins,
    //     numElems);
    //   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    //
    //   fillOutputValAndPositions<<<gridSize,blockSize>>>(d_outputVals,
    //     d_outputPos, d_inputVals, d_inputPos,
    //     d_cdfs + (i*numBins), d_relative_offsets,
    //     d_predicate + (i*numElems), numElems);
    //   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // }
    checkCudaErrors(cudaMemset((void *) d_relative_offsets, 0, elemBytes/numHistograms));
  }

  if(numHistograms%2==1)
  {
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  }

  checkCudaErrors(cudaFree(d_relative_offsets));
  checkCudaErrors(cudaFree(d_cdfs));
  checkCudaErrors(cudaFree(d_predicate));
  checkCudaErrors(cudaFree(d_mask_array));

}
