/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Main entry of dense matrix-matrix multiplication kernel
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include "parboil.h"
#include "sgemm_kernel.cu"

// I/O routines
bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

int
main (int argc, char *argv[]) {

  //struct pb_TimerSet timers;

  float *dA, *dB, *dC;
  size_t A_sz, B_sz, C_sz;
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  //pb_InitializeTimerSet(&timers);

  /* Initialize the parameters structure */
  struct pb_Parameters *params = (struct pb_Parameters *)malloc(sizeof(struct pb_Parameters));
  params->outFile = NULL;
  params->inpFiles = (char **)malloc(sizeof(char *)*3);
  params->inpFiles[0] = NULL;

  /* Read command line. Expect 3 inputs: A, B and B^T 
     in column-major layout*/
#ifdef SIZE0
  params->inpFiles[0] = "~/software/parboil-2.5/datasets/sgemm/small/input/matrix1.txt";
  params->inpFiles[1] = "~/software/parboil-2.5/datasets/sgemm/small/input/matrix2.txt";
  params->inpFiles[2] = "~/software/parboil-2.5/datasets/sgemm/small/input/matrix2t.txt";
#endif
#ifdef SIZE1
  params->inpFiles[0] = "~/software/parboil-2.5/datasets/sgemm/medium/input/matrix1.txt";
  params->inpFiles[1] = "~/software/parboil-2.5/datasets/sgemm/medium/input/matrix2.txt";
  params->inpFiles[2] = "~/software/parboil-2.5/datasets/sgemm/medium/input/matrix2t.txt";
#endif
  /* Read in data */
  //pb_SwitchToTimer(&timers, pb_TimerID_IO);

  // load A
  readColMajorMatrixFile(params->inpFiles[0],
      matArow, matAcol, matA);
  // copy A to device memory
  A_sz = matArow*matAcol*sizeof(float);

  // load B^T
  readColMajorMatrixFile(params->inpFiles[2],
      matBcol, matBrow, matBT);

 // pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  B_sz = matBrow*matBcol*sizeof(float);

  // allocate space for C
  C_sz = matArow*matBcol*sizeof(float);

  // CUDA memory allocation
  std::vector<float> matC(matArow*matBcol);
  cudaMalloc((void**)&dA, A_sz);
  cudaMalloc((void**)&dB, B_sz);
  cudaMalloc((void**)&dC, C_sz);
  
  // Copy A and B^T into device memory
  //pb_SwitchToTimer( &timers, pb_TimerID_COPY );
  cudaMemcpy(dA, &matA.front(), A_sz, cudaMemcpyHostToDevice); 
  cudaMemcpy(dB, &matBT.front(), B_sz, cudaMemcpyHostToDevice); 

  //pb_SwitchToTimer( &timers, pb_TimerID_KERNEL );

  // Use standard sgemm interface
  regtileSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, \
      dA, matArow, dB, matBcol, 0.0f, dC, matArow);

  if (params->outFile) {
    //pb_SwitchToTimer( &timers, pb_TimerID_COPY );
    cudaMemcpy(&matC.front(), dC, C_sz, cudaMemcpyDeviceToHost);
    /* Write C to file */
    //pb_SwitchToTimer(&timers, pb_TimerID_IO);
    writeColMajorMatrixFile(params->outFile,
	matArow, matBcol, matC); 
  }

  //pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  //double GPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_KERNEL]));
  //std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/GPUtime/1e9 << std::endl;
  //pb_PrintTimerSet(&timers);
  //pb_FreeParameters(params);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}

/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* I/O routines for reading and writing matrices in column-major
 * layout
 */

#include<fstream>
#include<iostream>
#include<vector>

bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v)
{
  std::cerr << "Opening file:"<< fn << std::endl;
  std::fstream f(fn, std::fstream::in);
  if ( !f.good() ) {
    return false;
  }

  // Read # of rows and cols
  f >> nr_row;
  f >> nr_col;

  float data;
  std::cerr << "Matrix dimension: "<<nr_row<<"x"<<nr_col<<std::endl;
  while (f.good() ) {
    f >> data;
    v.push_back(data);
  }
  v.pop_back(); // remove the duplicated last element

  return true;
}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col, std::vector<float>&v)
{
  std::cerr << "Opening file:"<< fn << " for write." << std::endl;
  std::fstream f(fn, std::fstream::out);
  if ( !f.good() ) {
    return false;
  }

  // Read # of rows and cols
  f << nr_row << " "<<nr_col<<" ";

  std::cerr << "Matrix dimension: "<<nr_row<<"x"<<nr_col<<std::endl;
  for (unsigned i = 0; i < v.size(); ++i) {
    f << v[i] << ' ';
  }
  f << "\n";
  return true;

}
