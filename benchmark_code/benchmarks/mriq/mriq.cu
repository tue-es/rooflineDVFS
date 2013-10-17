/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * C code for creating the Q data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <malloc.h>

#include "parboil.h"

#include "file.h"
#include "computeQ.cu"

void computePhiMag_GPU(int numK, float* phiR_d, float* phiI_d, float* phiMag_d)
{
  int phiMagBlocks = numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  if (numK % KERNEL_PHI_MAG_THREADS_PER_BLOCK)
    phiMagBlocks++;
  dim3 DimPhiMagBlock(KERNEL_PHI_MAG_THREADS_PER_BLOCK, 1);
  dim3 DimPhiMagGrid(phiMagBlocks, 1);

  ComputePhiMag_GPU <<< DimPhiMagGrid, DimPhiMagBlock >>> 
    (phiR_d, phiI_d, phiMag_d, numK);
}

void computeQ_GPU(int numK, int numX,
                  float* x_d, float* y_d, float* z_d,
                  kValues* kVals,
                  float* Qr_d, float* Qi_d)
{
  int QGrids = numK / KERNEL_Q_K_ELEMS_PER_GRID;
  if (numK % KERNEL_Q_K_ELEMS_PER_GRID)
    QGrids++;
  int QBlocks = numX / KERNEL_Q_THREADS_PER_BLOCK;
  if (numX % KERNEL_Q_THREADS_PER_BLOCK)
    QBlocks++;
  dim3 DimQBlock(KERNEL_Q_THREADS_PER_BLOCK, 1);
  dim3 DimQGrid(QBlocks, 1);

  for (int QGrid = 0; QGrid < QGrids; QGrid++) {
    // Put the tile of K values into constant mem
    int QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;
    kValues* kValsTile = kVals + QGridBase;
    int numElems = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase);

    cudaMemcpyToSymbol(ck, kValsTile, numElems * sizeof(kValues), 0);

    ComputeQ_GPU <<< DimQGrid, DimQBlock >>>
      (numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d);
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
   float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
}



static void
setupMemoryGPU(int num, int size, float*& dev_ptr, float*& host_ptr)
{
  cudaMalloc ((void **) &dev_ptr, num * size);
  CUDA_ERRCK;
  cudaMemcpy (dev_ptr, host_ptr, num * size, cudaMemcpyHostToDevice);
  CUDA_ERRCK;
}

static void
cleanupMemoryGPU(int num, int size, float *& dev_ptr, float * host_ptr)
{
  cudaMemcpy (host_ptr, dev_ptr, num * size, cudaMemcpyDeviceToHost);
  CUDA_ERRCK;
  cudaFree(dev_ptr);
  CUDA_ERRCK;
}

int
main (int argc, char *argv[]) {
  int numX, numK;		/* Number of X and K values */
  int original_numK;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  float *Qr, *Qi;		/* Q signal (complex) */

  struct kValues* kVals;

  //struct pb_Parameters *params;
  //struct pb_TimerSet timers;

  //pb_InitializeTimerSet(&timers);

  /* Read command line */
  //params = pb_ReadParameters(&argc, argv);
  
  /* Initialize the parameters structure */
  struct pb_Parameters *params = (struct pb_Parameters *)malloc(sizeof(struct pb_Parameters));
  params->outFile = NULL;
  params->inpFiles = (char **)malloc(sizeof(char *));
  params->inpFiles[0] = NULL;
  
// Read input from command line
#ifdef SIZE0
  params->inpFiles[0] = "~/software/parboil-2.5/datasets/mri-q/small/input/32_32_32_dataset.bin";
#endif
#ifdef SIZE1
  params->inpFiles[0] = "~/software/parboil-2.5/datasets/mri-q/large/input/64_64_64_dataset.bin";
#endif
  /*
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
    {
      fprintf(stderr, "Expecting one input filename\n");
      exit(-1);
    }
    */
  
  /* Read in data */
  //pb_SwitchToTimer(&timers, pb_TimerID_IO);
  inputData(params->inpFiles[0],
	    &original_numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
    numK = original_numK;
  else
    {
      int inputK;
      char *end;
      inputK = strtol(argv[1], &end, 10);
      if (end == argv[1])
	{
	  fprintf(stderr, "Expecting an integer parameter\n");
	  exit(-1);
	}

      numK = MIN(inputK, original_numK);
    }

  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
         numX, original_numK, numK);

  //pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  /* Create CPU data structures */
  createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);

  /* GPU section 1 (precompute PhiMag) */
  {
    /* Mirror several data structures on the device */
    float *phiR_d, *phiI_d;
    float *phiMag_d;

    //pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    setupMemoryGPU(numK, sizeof(float), phiR_d, phiR);
    setupMemoryGPU(numK, sizeof(float), phiI_d, phiI);
    cudaMalloc((void **)&phiMag_d, numK * sizeof(float));
    CUDA_ERRCK;

    cudaThreadSynchronize();
    //pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

    computePhiMag_GPU(numK, phiR_d, phiI_d, phiMag_d);

    cudaThreadSynchronize();
    //pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    cleanupMemoryGPU(numK, sizeof(float), phiMag_d, phiMag);
    cudaFree(phiR_d);
    cudaFree(phiI_d);
  }

  //pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  for (int k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  free(phiMag);

  /* GPU section 2 */
  {
    float *x_d, *y_d, *z_d;
    float *Qr_d, *Qi_d;

    //pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    setupMemoryGPU(numX, sizeof(float), x_d, x);
    setupMemoryGPU(numX, sizeof(float), y_d, y);
    setupMemoryGPU(numX, sizeof(float), z_d, z);
    cudaMalloc((void **)&Qr_d, numX * sizeof(float));
    CUDA_ERRCK;
    cudaMemset((void *)Qr_d, 0, numX * sizeof(float));
    cudaMalloc((void **)&Qi_d, numX * sizeof(float));
    CUDA_ERRCK;
    cudaMemset((void *)Qi_d, 0, numX * sizeof(float));

    cudaThreadSynchronize();
    //pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

    computeQ_GPU(numK, numX, x_d, y_d, z_d, kVals, Qr_d, Qi_d);

    cudaThreadSynchronize();
    //pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    cleanupMemoryGPU(numX, sizeof(float), Qr_d, Qr);
    cleanupMemoryGPU(numX, sizeof(float), Qi_d, Qi);
  }

  //pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  if (params->outFile)
    {
      /* Write Q to file */
      //pb_SwitchToTimer(&timers, pb_TimerID_IO);
      outputData(params->outFile, Qr, Qi, numX);
      //pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    }

  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (kVals);
  free (Qr);
  free (Qi);

  //pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  //pb_PrintTimerSet(&timers);

  //pb_FreeParameters(params);

  return 0;
}

void inputData(char* fName, int* _numK, int* _numX,
               float** kx, float** ky, float** kz,
               float** x, float** y, float** z,
               float** phiR, float** phiI)
{
  int numK, numX;
  FILE* fid = fopen(fName, "r");
  size_t temp;

  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open input file\n");
      exit(-1);
    }
  temp = fread (&numK, sizeof (int), 1, fid);
  *_numK = numK;
  temp = fread (&numX, sizeof (int), 1, fid);
  *_numX = numX;
  *kx = (float *) memalign(16, numK * sizeof (float));
  temp = fread (*kx, sizeof (float), numK, fid);
  *ky = (float *) memalign(16, numK * sizeof (float));
  temp = fread (*ky, sizeof (float), numK, fid);
  *kz = (float *) memalign(16, numK * sizeof (float));
  temp = fread (*kz, sizeof (float), numK, fid);
  *x = (float *) memalign(16, numX * sizeof (float));
  temp = fread (*x, sizeof (float), numX, fid);
  *y = (float *) memalign(16, numX * sizeof (float));
  temp = fread (*y, sizeof (float), numX, fid);
  *z = (float *) memalign(16, numX * sizeof (float));
  temp = fread (*z, sizeof (float), numX, fid);
  *phiR = (float *) memalign(16, numK * sizeof (float));
  temp = fread (*phiR, sizeof (float), numK, fid);
  *phiI = (float *) memalign(16, numK * sizeof (float));
  temp = fread (*phiI, sizeof (float), numK, fid);
  temp += numK;
  fclose (fid); 
}

void outputData(char* fName, float* outR, float* outI, int numX)
{
  FILE* fid = fopen(fName, "w");
  uint32_t tmp32;

  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open output file\n");
      exit(-1);
    }

  /* Write the data size */
  tmp32 = numX;
  fwrite(&tmp32, sizeof(uint32_t), 1, fid);

  /* Write the reconstructed data */
  fwrite (outR, sizeof (float), numX, fid);
  fwrite (outI, sizeof (float), numX, fid);
  fclose (fid);
}
