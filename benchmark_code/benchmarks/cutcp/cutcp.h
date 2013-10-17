/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef CUTOFF_H
#define CUTOFF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include "parboil.h"
#include "atom.h"

#define CELLEN      4.f
#define INV_CELLEN  (1.f/CELLEN)

#define LINELEN 96
#define INITLEN 20

#define ERRTOL 1e-4f

#define NOKERNELS             0
#define CUTOFF1               1
#define CUTOFF6              32
#define CUTOFF6OVERLAP       64
#define CUTOFFCPU         16384
 
#ifdef __DEVICE_EMULATION__
#define DEBUG
/* define which grid block and which thread to examine */
#define BX  0
#define BY  0
#define TX  0
#define TY  0
#define TZ  0
#define EMU(code) do { \
  if (blockIdx.x==BX && blockIdx.y==BY && \
      threadIdx.x==TX && threadIdx.y==TY && threadIdx.z==TZ) { \
    code; \
  } \
} while (0)
#define INT(n)    printf("%s = %d\n", #n, n)
#define FLOAT(f)  printf("%s = %g\n", #f, (double)(f))
#define INT3(n)   printf("%s = %d %d %d\n", #n, (n).x, (n).y, (n).z)
#define FLOAT4(f) printf("%s = %g %g %g %g\n", #f, (double)(f).x, \
    (double)(f).y, (double)(f).z, (double)(f).w)
#else
#define EMU(code)
#define INT(n)
#define FLOAT(f)
#define INT3(n)
#define FLOAT4(f)
#endif

/* report error from CUDA */
#define CUERR \
  do { \
    cudaError_t err; \
    if ((err = cudaGetLastError()) != cudaSuccess) { \
      printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
      return -1; \
    } \
  } while (0)

/*
 * neighbor list:
 * stored in constant memory as table of offsets
 * flat index addressing is computed by kernel
 *
 * reserve enough memory for 11^3 stencil of grid cells
 * this fits within 16K of memory
 */
#define NBRLIST_DIM  11
#define NBRLIST_MAXLEN (NBRLIST_DIM * NBRLIST_DIM * NBRLIST_DIM)
__constant__ int NbrListLen;
__constant__ int3 NbrList[NBRLIST_MAXLEN];

/*
 * atom bins cached into shared memory for processing
 *
 * this reserves 4K of shared memory for 32 atom bins each containing 8 atoms,
 * should permit scheduling of up to 3 thread blocks per SM
 */
#define BIN_DEPTH         8  /* max number of atoms per bin */
#define BIN_SIZE         32  /* size of bin in floats */
#define BIN_CACHE_MAXLEN 32  /* max number of atom bins to cache */

#define BIN_LENGTH      4.f  /* spatial length in Angstroms */
#define BIN_INVLEN  (1.f / BIN_LENGTH)
/* assuming density of 1 atom / 10 A^3, expectation is 6.4 atoms per bin
 * so that bin fill should be 80% (for non-empty regions of space) */

#define REGION_SIZE     512  /* number of floats in lattice region */
#define SUB_REGION_SIZE 128  /* number of floats in lattice sub-region */


#ifdef __cplusplus
extern "C" {
#endif

#define SHIFTED

  /* A structure to record how points in 3D space map to array
     elements.  Array element (z, y, x)
     where 0 <= x < nx, 0 <= y < ny, 0 <= z < nz
     maps to coordinate (xlo, ylo, zlo) + h * (x, y, z).
  */
  typedef struct LatticeDim_t {
    /* Number of lattice points in x, y, z dimensions */
    int nx, ny, nz;

    /* Lowest corner of lattice */
    Vec3 lo;

    /* Lattice spacing */
    float h;
  } LatticeDim;

  /* An electric potential field sampled on a regular grid.  The
     lattice size and grid point positions are specified by 'dim'.
  */
  typedef struct Lattice_t {
    LatticeDim dim;
    float *lattice;
  } Lattice;

  LatticeDim lattice_from_bounding_box(Vec3 lo, Vec3 hi, float h);

  Lattice *create_lattice(LatticeDim dim);
  void destroy_lattice(Lattice *);

  int gpu_compute_cutoff_potential_lattice(
      Lattice *lattice,
      float cutoff,                      /* cutoff distance */
      Atoms *atom,                       /* array of atoms */
      int verbose                        /* print info/debug messages */
    );

  int cpu_compute_cutoff_potential_lattice(
      Lattice *lattice,                  /* the lattice */
      float cutoff,                      /* cutoff distance */
      Atoms *atoms                       /* array of atoms */
    );

  int remove_exclusions(
      Lattice *lattice,                  /* the lattice */
      float exclcutoff,                  /* exclusion cutoff distance */
      Atoms *atom                        /* array of atoms */
    );

#ifdef __cplusplus
}
#endif


#include "output.h"

#endif /* CUTOFF_H */
