/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

 
#include "cutcp.h"
 
#define ITERS 2

/*
 * potential lattice is decomposed into size 8^3 lattice point "regions"
 *
 * THIS IMPLEMENTATION:  one thread per lattice point
 * thread block size 128 gives 4 thread blocks per region
 * kernel is invoked for each x-y plane of regions,
 * where gridDim.x is 4*(x region dimension) so that blockIdx.x 
 * can absorb the z sub-region index in its 2 lowest order bits
 *
 * Regions are stored contiguously in memory in row-major order
 *
 * The bins have to not only cover the region, but they need to surround
 * the outer edges so that region sides and corners can still use
 * neighbor list stencil.  The binZeroAddr is actually a shifted pointer into
 * the bin array (binZeroAddr = binBaseAddr + (c*binDim_y + c)*binDim_x + c)
 * where c = ceil(cutoff / binsize).  This allows for negative offsets to
 * be added to myBinIndex.
 *
 * The (0,0,0) spatial origin corresponds to lower left corner of both
 * regionZeroAddr and binZeroAddr.  The atom coordinates are translated
 * during binning to enforce this assumption.
 */
__global__ static void cuda_cutoff_potential_lattice(
    int binDim_x,
    int binDim_y,
    float4 *binZeroAddr,    /* address of atom bins starting at origin */
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    float *regionZeroAddr,  /* address of lattice regions starting at origin */
    int zRegionIndex
    )
{
  __shared__ float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
  __shared__ float *mySubRegionAddr;
  __shared__ int3 myBinIndex;

  //const int xRegionIndex = (blockIdx.x >> 2);
  //const int yRegionIndex = blockIdx.y;

  /* thread id */
  const int tid = (threadIdx.z*8 + threadIdx.y)*8 + threadIdx.x;

  /* neighbor index */
  int nbrid;

  /* this is the start of the sub-region indexed by tid */
  mySubRegionAddr = regionZeroAddr + ((zRegionIndex*gridDim.y
        + blockIdx.y)*(gridDim.x>>2) + (blockIdx.x >> 2))*REGION_SIZE
        + (blockIdx.x&3)*SUB_REGION_SIZE;

  /* spatial coordinate of this lattice point */
  float x = (8 * (blockIdx.x >> 2) + threadIdx.x) * h;
  float y = (8 * blockIdx.y + threadIdx.y) * h;
  float z = (8 * zRegionIndex + 2*(blockIdx.x&3) + threadIdx.z) * h;

  int totalbins = 0;
  int numbins;

  /* bin number determined by center of region */
  myBinIndex.x = (int) floorf((8 * (blockIdx.x >> 2) + 4) * h * BIN_INVLEN);
  myBinIndex.y = (int) floorf((8 * blockIdx.y + 4) * h * BIN_INVLEN);
  myBinIndex.z = (int) floorf((8 * zRegionIndex + 4) * h * BIN_INVLEN);

  /* first neighbor in list for me to cache */
  nbrid = (tid >> 4);

  numbins = BIN_CACHE_MAXLEN;

  float energy = 0.f;
  for (totalbins = 0;  totalbins < NbrListLen;  totalbins += numbins) {
    int bincnt;

    /* start of where to write in shared memory */
    int startoff = BIN_SIZE * (tid >> 4);

    /* each half-warp to cache up to 4 atom bins */
    for (bincnt = 0;  bincnt < 4 && nbrid < NbrListLen;  bincnt++, nbrid += 8) {
      int i = myBinIndex.x + NbrList[nbrid].x;
      int j = myBinIndex.y + NbrList[nbrid].y;
      int k = myBinIndex.z + NbrList[nbrid].z;

      /* determine global memory location of atom bin */
      float *p_global = ((float *) binZeroAddr)
        + (((k*binDim_y) + j)*binDim_x + i) * BIN_SIZE;

      /* coalesced read from global memory -
       * retain same ordering in shared memory for now */
      int tidmask = tid & 15;
      int binIndex = startoff + bincnt*8*BIN_SIZE;

      AtomBinCache[binIndex + tidmask   ] = p_global[tidmask   ];
      AtomBinCache[binIndex + tidmask+16] = p_global[tidmask+16];
    }
    __syncthreads();

    /* no warp divergence */
    if (totalbins + BIN_CACHE_MAXLEN > NbrListLen) {
      numbins = NbrListLen - totalbins;
    }

    for (bincnt = 0;  bincnt < numbins;  bincnt++) {
      int i;
      float r2;

      for (i = 0;  i < BIN_DEPTH;  i++) {
        float ax = AtomBinCache[bincnt * BIN_SIZE + i*4];
        float ay = AtomBinCache[bincnt * BIN_SIZE + i*4 + 1];
        float az = AtomBinCache[bincnt * BIN_SIZE + i*4 + 2];
        float aq = AtomBinCache[bincnt * BIN_SIZE + i*4 + 3];
        if (0.f == aq) break;  /* no more atoms in bin */
        r2 = (ax - x) * (ax - x) + (ay - y) * (ay - y) + (az - z) * (az - z);
        if (r2 < cutoff2) {
          float s = (1.f - r2 * inv_cutoff2);
          energy += aq * rsqrtf(r2) * s * s;
        }
      } /* end loop over atoms in bin */
    } /* end loop over cached atom bins */
    __syncthreads();

  } /* end loop over neighbor list */

  /* store into global memory */
  mySubRegionAddr[tid] = energy;
}




int gpu_compute_cutoff_potential_lattice(
    Lattice *lattice,                  /* the lattice */
    float cutoff,                      /* cutoff distance */
    Atoms *atoms,                      /* array of atoms */
    int verbose                        /* print info/debug messages */
    )
{
  int nx = lattice->dim.nx;
  int ny = lattice->dim.ny;
  int nz = lattice->dim.nz;
  float xlo = lattice->dim.lo.x;
  float ylo = lattice->dim.lo.y;
  float zlo = lattice->dim.lo.z;
  float h = lattice->dim.h;
  int natoms = atoms->size;
  Atom *atom = atoms->atoms;

  int3 nbrlist[NBRLIST_MAXLEN];
  int nbrlistlen = 0;

  int binHistoFull[BIN_DEPTH+1] = { 0 };   /* clear every array element */
  int binHistoCover[BIN_DEPTH+1] = { 0 };  /* clear every array element */
  int num_excluded = 0;

  int xRegionDim, yRegionDim, zRegionDim;
  int xRegionIndex, yRegionIndex, zRegionIndex;
  int xOffset, yOffset, zOffset;
  int lnx, lny, lnz, lnall;
  float *regionZeroAddr, *thisRegion;
  float *regionZeroCuda;
  int index, indexRegion;

  int c;
  int3 binDim;
  int nbins;
  float4 *binBaseAddr, *binZeroAddr;
  float4 *binBaseCuda, *binZeroCuda;
  int *bincntBaseAddr, *bincntZeroAddr;
  Atoms *extra = NULL;

  int i, j, k, n;
  int sum, total;

  float avgFillFull, avgFillCover;
  const float cutoff2 = cutoff * cutoff;
  const float inv_cutoff2 = 1.f / cutoff2;

  dim3 gridDim, blockDim;

  // Caller has made the 'compute' timer active

  /* pad lattice to be factor of 8 in each dimension */
  xRegionDim = (int) ceilf(nx/8.f);
  yRegionDim = (int) ceilf(ny/8.f);
  zRegionDim = (int) ceilf(nz/8.f);

  lnx = 8 * xRegionDim;
  lny = 8 * yRegionDim;
  lnz = 8 * zRegionDim;
  lnall = lnx * lny * lnz;

  /* will receive energies from CUDA */
  regionZeroAddr = (float *) malloc(lnall * sizeof(float));

  /* create bins */
  c = (int) ceil(cutoff * BIN_INVLEN);  /* count extra bins around lattice */
  binDim.x = (int) ceil(lnx * h * BIN_INVLEN) + 2*c;
  binDim.y = (int) ceil(lny * h * BIN_INVLEN) + 2*c;
  binDim.z = (int) ceil(lnz * h * BIN_INVLEN) + 2*c;
  nbins = binDim.x * binDim.y * binDim.z;
  binBaseAddr = (float4 *) calloc(nbins * BIN_DEPTH, sizeof(float4));
  binZeroAddr = binBaseAddr + ((c * binDim.y + c) * binDim.x + c) * BIN_DEPTH;

  bincntBaseAddr = (int *) calloc(nbins, sizeof(int));
  bincntZeroAddr = bincntBaseAddr + (c * binDim.y + c) * binDim.x + c;

  /* create neighbor list */
  if (ceilf(BIN_LENGTH / (8*h)) == floorf(BIN_LENGTH / (8*h))) {
    float s = sqrtf(3);
    float r2 = (cutoff + s*BIN_LENGTH) * (cutoff + s*BIN_LENGTH);
    int cnt = 0;
    /* develop neighbor list around 1 cell */
    if (2*c + 1 > NBRLIST_DIM) {
      fprintf(stderr, "must have cutoff <= %f\n",
          (NBRLIST_DIM-1)/2 * BIN_LENGTH);
      return -1;
    }
    for (k = -c;  k <= c;  k++) {
      for (j = -c;  j <= c;  j++) {
        for (i = -c;  i <= c;  i++) {
          if ((i*i + j*j + k*k)*BIN_LENGTH*BIN_LENGTH >= r2) continue;
          nbrlist[cnt].x = i;
          nbrlist[cnt].y = j;
          nbrlist[cnt].z = k;
          cnt++;
        }
      }
    }
    nbrlistlen = cnt;
  }
  else if (8*h <= 2*BIN_LENGTH) {
    float s = 2.f*sqrtf(3);
    float r2 = (cutoff + s*BIN_LENGTH) * (cutoff + s*BIN_LENGTH);
    int cnt = 0;
    /* develop neighbor list around 3-cube of cells */
    if (2*c + 3 > NBRLIST_DIM) {
      fprintf(stderr, "must have cutoff <= %f\n",
          (NBRLIST_DIM-3)/2 * BIN_LENGTH);
      return -1;
    }
    for (k = -c;  k <= c;  k++) {
      for (j = -c;  j <= c;  j++) {
        for (i = -c;  i <= c;  i++) {
          if ((i*i + j*j + k*k)*BIN_LENGTH*BIN_LENGTH >= r2) continue;
          nbrlist[cnt].x = i;
          nbrlist[cnt].y = j;
          nbrlist[cnt].z = k;
          cnt++;
        }
      }
    }
    nbrlistlen = cnt;
  }
  else {
    fprintf(stderr, "must have h <= %f\n", 0.25 * BIN_LENGTH);
    return -1;
  }

  /* perform geometric hashing of atoms into bins */
  {
    /* array of extra atoms, permit average of one extra per bin */
    Atom *extra_atoms = (Atom *) calloc(nbins, sizeof(Atom));
    int extra_len = 0;
    
    for (n = 0;  n < natoms;  n++) {
      float4 p;
      p.x = atom[n].x - xlo;
      p.y = atom[n].y - ylo;
      p.z = atom[n].z - zlo;
      p.w = atom[n].q;
      i = (int) floorf(p.x * BIN_INVLEN);
      j = (int) floorf(p.y * BIN_INVLEN);
      k = (int) floorf(p.z * BIN_INVLEN);
      if (i >= -c && i < binDim.x - c &&
	  j >= -c && j < binDim.y - c &&
	  k >= -c && k < binDim.z - c &&
	  atom[n].q != 0) {
	int index = (k * binDim.y + j) * binDim.x + i;
	float4 *bin = binZeroAddr + index * BIN_DEPTH;
	int bindex = bincntZeroAddr[index];
	if (bindex < BIN_DEPTH) {
	  /* copy atom into bin and increase counter for this bin */
	  bin[bindex] = p;
	  bincntZeroAddr[index]++;
	}
	else {
	  /* add index to array of extra atoms to be computed with CPU */
	  if (extra_len >= nbins) {
	    fprintf(stderr, "exceeded space for storing extra atoms\n");
	    return -1;
	  }
	  extra_atoms[extra_len] = atom[n];
	  extra_len++;
	}
      }
      else {
	/* excluded atoms are either outside bins or neutrally charged */
	num_excluded++;
      }
    }

    /* Save result */
    extra = (Atoms *)malloc(sizeof(Atoms));
    extra->atoms = extra_atoms;
    extra->size = extra_len;
  }

  /* bin stats */
  sum = total = 0;
  for (n = 0;  n < nbins;  n++) {
    binHistoFull[ bincntBaseAddr[n] ]++;
    sum += bincntBaseAddr[n];
    total += BIN_DEPTH;
  }
  avgFillFull = sum / (float) total;
  sum = total = 0;
  for (k = 0;  k < binDim.z - 2*c;  k++) {
    for (j = 0;  j < binDim.y - 2*c;  j++) {
      for (i = 0;  i < binDim.x - 2*c;  i++) {
        int index = (k * binDim.y + j) * binDim.x + i;
        binHistoCover[ bincntZeroAddr[index] ]++;
        sum += bincntZeroAddr[index];
        total += BIN_DEPTH;
      }
    }
  }
  avgFillCover = sum / (float) total;

  if (verbose) {
    /* report */
    printf("number of atoms = %d\n", natoms);
    printf("lattice spacing = %g\n", h);
    printf("cutoff distance = %g\n", cutoff);
    printf("\n");
    printf("requested lattice dimensions = %d %d %d\n", nx, ny, nz);
    printf("requested space dimensions = %g %g %g\n", nx*h, ny*h, nz*h);
    printf("expanded lattice dimensions = %d %d %d\n", lnx, lny, lnz);
    printf("expanded space dimensions = %g %g %g\n", lnx*h, lny*h, lnz*h);
    printf("number of bytes for lattice data = %u\n", (unsigned int)(lnall*sizeof(float)));
    printf("\n");
    printf("bin padding thickness = %d\n", c);
    printf("bin cover dimensions = %d %d %d\n",
        binDim.x - 2*c, binDim.y - 2*c, binDim.z - 2*c);
    printf("bin full dimensions = %d %d %d\n", binDim.x, binDim.y, binDim.z);
    printf("number of bins = %d\n", nbins);
    printf("total number of atom slots = %d\n", nbins * BIN_DEPTH);
    printf("%% overhead space = %g\n",
        (natoms / (double) (nbins * BIN_DEPTH)) * 100);
    printf("number of bytes for bin data = %u\n",
        (unsigned int)(nbins * BIN_DEPTH * sizeof(float4)));
    printf("\n");
    printf("bin histogram with padding:\n");
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      printf("     number of bins with %d atoms:  %d\n", n, binHistoFull[n]);
      sum += binHistoFull[n];
    }
    printf("     total number of bins:  %d\n", sum);
    printf("     %% average fill:  %g\n", avgFillFull * 100);
    printf("\n");
    printf("bin histogram excluding padding:\n");
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      printf("     number of bins with %d atoms:  %d\n", n, binHistoCover[n]);
      sum += binHistoCover[n];
    }
    printf("     total number of bins:  %d\n", sum);
    printf("     %% average fill:  %g\n", avgFillCover * 100);
    printf("\n");
    printf("number of extra atoms = %d\n", extra->size);
    printf("%% atoms that are extra = %g\n", (extra->size / (double) natoms) * 100);
    printf("\n");

    /* sanity check on bins */
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      sum += n * binHistoFull[n];
    }
    sum += extra->size + num_excluded;
    printf("sanity check on bin histogram with edges:  "
        "sum + others = %d\n", sum);
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      sum += n * binHistoCover[n];
    }
    sum += extra->size + num_excluded;
    printf("sanity check on bin histogram excluding edges:  "
        "sum + others = %d\n", sum);
    printf("\n");

    /* neighbor list */
    printf("neighbor list length = %d\n", nbrlistlen);
    printf("\n");
  }

  /* setup CUDA kernel parameters */
  gridDim.x = 4 * xRegionDim;
  gridDim.y = yRegionDim;
  gridDim.z = 1;
  blockDim.x = 8;
  blockDim.y = 8;
  blockDim.z = 2;

  /* allocate and initialize memory on CUDA device */
  //pb_SwitchToTimer(timers, pb_TimerID_COPY);
  if (verbose) {
    printf("Allocating %.2fMB on CUDA device for potentials\n",
           lnall * sizeof(float) / (double) (1024*1024));
  }
  cudaMalloc((void **) &regionZeroCuda, lnall * sizeof(float));
  CUERR;
  cudaMemset(regionZeroCuda, 0, lnall * sizeof(float));
  CUERR;
  if (verbose) {
    printf("Allocating %.2fMB on CUDA device for atom bins\n",
           nbins * BIN_DEPTH * sizeof(float4) / (double) (1024*1024));
  }
  cudaMalloc((void **) &binBaseCuda, nbins * BIN_DEPTH * sizeof(float4));
  CUERR;
  cudaMemcpy(binBaseCuda, binBaseAddr, nbins * BIN_DEPTH * sizeof(float4),
      cudaMemcpyHostToDevice);
  CUERR;
  binZeroCuda = binBaseCuda + ((c * binDim.y + c) * binDim.x + c) * BIN_DEPTH;
  cudaMemcpyToSymbol(NbrListLen, &nbrlistlen, sizeof(int), 0);
  CUERR;
  cudaMemcpyToSymbol(NbrList, nbrlist, nbrlistlen * sizeof(int3), 0);
  CUERR;

  if (verbose) 
    printf("\n");

  /* loop over z-dimension, invoke CUDA kernel for each x-y plane */
  //pb_SwitchToTimer(timers, pb_TimerID_KERNEL);
  zRegionDim = ITERS;
  printf("Invoking CUDA kernel on %d region planes...\n", zRegionDim);
  for (zRegionIndex = 0;  zRegionIndex < zRegionDim;  zRegionIndex++) {
    printf("  computing plane %d\r", zRegionIndex);
    fflush(stdout);
    cuda_cutoff_potential_lattice<<<gridDim, blockDim, 0>>>(binDim.x, binDim.y,
        binZeroCuda, h, cutoff2, inv_cutoff2, regionZeroCuda, zRegionIndex);
    CUERR;
  }
  printf("Finished CUDA kernel calls                        \n");

  /* copy result regions from CUDA device */
  //pb_SwitchToTimer(timers, pb_TimerID_COPY);
  cudaMemcpy(regionZeroAddr, regionZeroCuda, lnall * sizeof(float),
      cudaMemcpyDeviceToHost);
  CUERR;

  /* free CUDA memory allocations */
  cudaFree(regionZeroCuda);
  cudaFree(binBaseCuda);

  /* transpose regions back into lattice */
  //pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);
  for (k = 0;  k < nz;  k++) {
    zRegionIndex = (k >> 3);
    zOffset = (k & 7);

    for (j = 0;  j < ny;  j++) {
      yRegionIndex = (j >> 3);
      yOffset = (j & 7);

      for (i = 0;  i < nx;  i++) {
        xRegionIndex = (i >> 3);
        xOffset = (i & 7);

        thisRegion = regionZeroAddr
          + ((zRegionIndex * yRegionDim + yRegionIndex) * xRegionDim
              + xRegionIndex) * REGION_SIZE;

        indexRegion = (zOffset * 8 + yOffset) * 8 + xOffset;
        index = (k * ny + j) * nx + i;

        lattice->lattice[index] = thisRegion[indexRegion];
      }
    }
  }

  /* handle extra atoms */
  if (extra->size > 0) {
    printf("computing extra atoms on CPU\n");
    if (cpu_compute_cutoff_potential_lattice(lattice, cutoff, extra)) {
      fprintf(stderr, "cpu_compute_cutoff_potential_lattice() failed "
          "for extra atoms\n");
      return -1;
    }
    printf("\n");
  }

  /* cleanup memory allocations */
  free(regionZeroAddr);
  free(binBaseAddr);
  free(bincntBaseAddr);
  free_atom(extra);

  return 0;
}

int appenddata(const char *filename, int size, double time) {
  FILE *fp;
  fp=fopen(filename, "a");
  if (fp == NULL) {
    printf("error appending to file %s..\n", filename);
    return -1;
  }
  fprintf(fp, "%d  %.3f\n", size, time);
  fclose(fp);
  return 0;
}

LatticeDim
lattice_from_bounding_box(Vec3 lo, Vec3 hi, float h)
{
  LatticeDim ret;

  ret.nx = (int) floorf((hi.x-lo.x)/h) + 1;
  ret.ny = (int) floorf((hi.y-lo.y)/h) + 1;
  ret.nz = (int) floorf((hi.z-lo.z)/h) + 1;
  ret.lo = lo;
  ret.h = h;

  return ret;
}

Lattice *
create_lattice(LatticeDim dim)
{
  int size;
  Lattice *lat = (Lattice *)malloc(sizeof(Lattice));

  if (lat == NULL) {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  lat->dim = dim;

  /* Round up the allocated size to a multiple of 8 */
  size = ((dim.nx * dim.ny * dim.nz) + 7) & ~7;
  lat->lattice = (float *)calloc(size, sizeof(float));

  if (lat->lattice == NULL) {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  return lat;
}


void
destroy_lattice(Lattice *lat)
{
  if (lat) {
    free(lat->lattice);
    free(lat);
  }
}

int main(int argc, char *argv[]) {
  Atoms *atom;

  LatticeDim lattice_dim;
  Lattice *gpu_lattice;
  Vec3 min_ext, max_ext;  /* Bounding box of atoms */
  Vec3 lo, hi;      /* Bounding box with padding  */

  float h = 0.5f;   /* Lattice spacing */
  float cutoff = 12.f;    /* Cutoff radius */
  float exclcutoff = 1.f; /* Radius for exclusion */
  float padding = 0.5f;   /* Bounding box padding distance */


  //struct pb_Parameters *parameters;
  //struct pb_TimerSet timers;
  
  /* Initialize the parameters structure */
  struct pb_Parameters *parameters = (struct pb_Parameters *)malloc(sizeof(struct pb_Parameters));
  parameters->outFile = NULL;
  parameters->inpFiles = (char **)malloc(sizeof(char *));
  parameters->inpFiles[0] = NULL;
  
// Read input from command line 
#ifdef SIZE0
  parameters->inpFiles[0] = "~/software/parboil-2.5/datasets/cutcp/small/input/watbox.sl40.pqr";
#endif
#ifdef SIZE1
  parameters->inpFiles[0] = "~/software/parboil-2.5/datasets/cutcp/large/input/watbox.sl100.pqr";
#endif

  /* Read input parameters */
  //parameters = pb_ReadParameters(&argc, argv);
  if (parameters == NULL) {
    exit(1);
  }

  /* Expect one input file */
  /*
  if (pb_Parameters_CountInputs(parameters) != 1) {
    fprintf(stderr, "Expecting one input file\n");
    exit(1);
  }*/

  //pb_InitializeTimerSet(&timers);
  //pb_SwitchToTimer(&timers, pb_TimerID_IO);

  {
    const char *pqrfilename = parameters->inpFiles[0];

    if (!(atom = read_atom_file(pqrfilename))) {
      fprintf(stderr, "read_atom_file() failed\n");
      exit(1);
    }
    printf("read %d atoms from file '%s'\n", atom->size, pqrfilename);
  }

  /* find extent of domain */
  //pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  get_atom_extent(&min_ext, &max_ext, atom);
  printf("extent of domain is:\n");
  printf("  minimum %g %g %g\n", min_ext.x, min_ext.y, min_ext.z);
  printf("  maximum %g %g %g\n", max_ext.x, max_ext.y, max_ext.z);

  printf("padding domain by %g Angstroms\n", padding);
  lo = (Vec3) {min_ext.x - padding, min_ext.y - padding, min_ext.z - padding};
  hi = (Vec3) {max_ext.x + padding, max_ext.y + padding, max_ext.z + padding};
  printf("domain lengths are %g by %g by %g\n", hi.x-lo.x, hi.y-lo.y, hi.z-lo.z);

  lattice_dim = lattice_from_bounding_box(lo, hi, h);
  gpu_lattice = create_lattice(lattice_dim);
  printf("\n");

  /*
   *  Run CUDA kernel
   *  (enter and exit with the 'compute' timer active)
   */
  if (gpu_compute_cutoff_potential_lattice(gpu_lattice, cutoff, atom, 0)) {
    fprintf(stderr, "Computation failed\n");
    exit(1);
  }

  /*
   * Zero the lattice points that are too close to an atom.  This is
   * necessary for numerical stability.
   */
  if (remove_exclusions(gpu_lattice, exclcutoff, atom)) {
    fprintf(stderr, "remove_exclusions() failed for gpu lattice\n");
    exit(1);
  }

  printf("\n");

  //pb_SwitchToTimer(&timers, pb_TimerID_IO);

  /* Print output */
  if (parameters->outFile) {
    write_lattice_summary(parameters->outFile, gpu_lattice);
  }
  //pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  /* Cleanup */
  destroy_lattice(gpu_lattice);
  free_atom(atom);

  //pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  //pb_PrintTimerSet(&timers);
  //pb_FreeParameters(parameters);

  return 0;
}



Atoms *read_atom_file(const char *fname)
{
  FILE *file;
  char line[LINELEN];

  Atom *atom;     /* Atom array */
  int len = INITLEN;    /* Size of atom array */
  int cnt = 0;      /* Number of atoms read */

  /* open atom "pqr" file */
  file = fopen(fname, "r");
  if (NULL==file) {
    fprintf(stderr, "can't open file \"%s\" for reading\n", fname);
    return NULL;
  }

  /* allocate initial atom array */
  atom = (Atom *) malloc(len * sizeof(Atom));
  if (NULL==atom) {
    fprintf(stderr, "can't allocate memory\n");
    return NULL;
  }

  /* loop to read pqr file line by line */
  while (fgets(line, LINELEN, file) != NULL) {

    if (strncmp(line, "ATOM  ", 6) != 0 && strncmp(line, "HETATM", 6) != 0) {
      continue;  /* skip anything that isn't an atom record */
    }

    if (cnt==len) {  /* extend atom array */
      void *tmp = realloc(atom, 2*len*sizeof(Atom));
      if (NULL==tmp) {
        fprintf(stderr, "can't allocate more memory\n");
        return NULL;
      }
      atom = (Atom *) tmp;
      len *= 2;
    }

    /* read position coordinates and charge from atom record */
    if (sscanf(line, "%*s %*d %*s %*s %*d %f %f %f %f", &(atom[cnt].x),
          &(atom[cnt].y), &(atom[cnt].z), &(atom[cnt].q)) != 4) {
      fprintf(stderr, "atom record %d does not have expected format\n", cnt+1);
      return NULL;
    }

    cnt++;  /* count atoms as we store them */
  }

  /* verify EOF and close file */
  if ( !feof(file) ) {
    fprintf(stderr, "did not find EOF\n");
    return NULL;
  }
  if (fclose(file)) {
    fprintf(stderr, "can't close file\n");
    return NULL;
  }

  /* Build the output data structure */
  {
    Atoms *out = (Atoms *)malloc(sizeof(Atoms));

    if (NULL == out) {
      fprintf(stderr, "can't allocate memory\n");
      return NULL;
    }

    out->size = cnt;
    out->atoms = atom;

    return out;
  }
}


void free_atom(Atoms *atom)
{
  if (atom) {
    free(atom->atoms);
    free(atom);
  }
}

void
get_atom_extent(Vec3 *out_lo, Vec3 *out_hi, Atoms *atom)
{
  Atom *atoms = atom->atoms;
  int natoms = atom->size;
  Vec3 lo;
  Vec3 hi;
  int n;

  hi.x = lo.x = atoms[0].x;
  hi.y = lo.y = atoms[0].y;
  hi.z = lo.z = atoms[0].z;

  for (n = 1; n < natoms; n++) {
    lo.x = fminf(lo.x, atoms[n].x);
    hi.x = fmaxf(hi.x, atoms[n].x);
    lo.y = fminf(lo.y, atoms[n].y);
    hi.y = fmaxf(hi.y, atoms[n].y);
    lo.z = fminf(lo.z, atoms[n].z);
    hi.z = fmaxf(hi.z, atoms[n].z);
  }

  *out_lo = lo;
  *out_hi = hi;
}



int remove_exclusions(
    Lattice *lattice,                  /* the lattice */
    float cutoff,                      /* exclusion cutoff distance */
    Atoms *atoms                       /* array of atoms */
    )
{
  int nx = lattice->dim.nx;
  int ny = lattice->dim.ny;
  int nz = lattice->dim.nz;
  float xlo = lattice->dim.lo.x;
  float ylo = lattice->dim.lo.y;
  float zlo = lattice->dim.lo.z;
  float gridspacing = lattice->dim.h;
  Atom *atom = atoms->atoms;

  const float a2 = cutoff * cutoff;
  const float inv_gridspacing = 1.f / gridspacing;
  const int radius = (int) ceilf(cutoff * inv_gridspacing) - 1;
    /* lattice point radius about each atom */

  int n;
  int i, j, k;
  int ia, ib, ic;
  int ja, jb, jc;
  int ka, kb, kc;
  int index;
  int koff, jkoff;

  float x, y, z;
  float dx, dy, dz;
  float dz2, dydz2, r2;
  float xstart, ystart;

  float *pg;

  int gindex;
  int ncell, nxcell, nycell, nzcell;
  int *first, *next;
  float inv_cellen = INV_CELLEN;
  Vec3 minext, maxext;

  /* find min and max extent */
  get_atom_extent(&minext, &maxext, atoms);

  /* number of cells in each dimension */
  nxcell = (int) floorf((maxext.x-minext.x) * inv_cellen) + 1;
  nycell = (int) floorf((maxext.y-minext.y) * inv_cellen) + 1;
  nzcell = (int) floorf((maxext.z-minext.z) * inv_cellen) + 1;
  ncell = nxcell * nycell * nzcell;

  /* allocate for cursor link list implementation */
  first = (int *) malloc(ncell * sizeof(int));
  for (gindex = 0;  gindex < ncell;  gindex++) {
    first[gindex] = -1;
  }
  next = (int *) malloc(atoms->size * sizeof(int));
  for (n = 0;  n < atoms->size;  n++) {
    next[n] = -1;
  }

  /* geometric hashing */
  for (n = 0;  n < atoms->size;  n++) {
    if (0==atom[n].q) continue;  /* skip any non-contributing atoms */
    i = (int) floorf((atom[n].x - minext.x) * inv_cellen);
    j = (int) floorf((atom[n].y - minext.y) * inv_cellen);
    k = (int) floorf((atom[n].z - minext.z) * inv_cellen);
    gindex = (k*nycell + j)*nxcell + i;
    next[n] = first[gindex];
    first[gindex] = n;
  }

  /* traverse the grid cells */
  for (gindex = 0;  gindex < ncell;  gindex++) {
    for (n = first[gindex];  n != -1;  n = next[n]) {
      x = atom[n].x - xlo;
      y = atom[n].y - ylo;
      z = atom[n].z - zlo;
      //q = atom[n].q;

      /* find closest grid point with position less than or equal to atom */
      ic = (int) (x * inv_gridspacing);
      jc = (int) (y * inv_gridspacing);
      kc = (int) (z * inv_gridspacing);

      /* find extent of surrounding box of grid points */
      ia = ic - radius;
      ib = ic + radius + 1;
      ja = jc - radius;
      jb = jc + radius + 1;
      ka = kc - radius;
      kb = kc + radius + 1;

      /* trim box edges so that they are within grid point lattice */
      if (ia < 0)   ia = 0;
      if (ib >= nx) ib = nx-1;
      if (ja < 0)   ja = 0;
      if (jb >= ny) jb = ny-1;
      if (ka < 0)   ka = 0;
      if (kb >= nz) kb = nz-1;

      /* loop over surrounding grid points */
      xstart = ia*gridspacing - x;
      ystart = ja*gridspacing - y;
      dz = ka*gridspacing - z;
      for (k = ka;  k <= kb;  k++, dz += gridspacing) {
        koff = k*ny;
        dz2 = dz*dz;

        dy = ystart;
        for (j = ja;  j <= jb;  j++, dy += gridspacing) {
          jkoff = (koff + j)*nx;
          dydz2 = dy*dy + dz2;

          dx = xstart;
          index = jkoff + ia;
          pg = lattice->lattice + index;

          for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
            r2 = dx*dx + dydz2;

      /* If atom and lattice point are too close, set the lattice value
       * to zero */
            if (r2 < a2) *pg = 0;
          }
        }
      } /* end loop over surrounding grid points */

    } /* end loop over atoms in a gridcell */
  } /* end loop over gridcells */

  /* free memory */
  free(next);
  free(first);

  return 0;
}

void
write_lattice_summary(const char *filename, Lattice *lattice)
{
  float *lattice_data = lattice->lattice;
  int nx = lattice->dim.nx;
  int ny = lattice->dim.ny;
  int nz = lattice->dim.nz;

  /* Open output file */
  FILE *outfile = fopen(filename, "w");

  if (outfile == NULL) {
    fprintf(stderr, "Cannot open output file\n");
    exit(1);
  }

  /* Write the sum of the the absolute values of all lattice potentials */
  {
    double abspotential = 0.0;
    float tmp;
    int i;

    for (i = 0; i < nx * ny * nz; i++)
      abspotential += fabs((double) lattice_data[i]);

    tmp = (float) abspotential;

    fwrite(&tmp, 1, sizeof(float), outfile);
  }

  /* Write the size of a lattice plane */
  {
    uint32_t tmp;

    tmp = (uint32_t) (lattice->dim.nx * lattice->dim.ny);
    fwrite(&tmp, 1, sizeof(uint32_t), outfile);
  }

  /* Write the plane of lattice data at z=0 and z = nz-1 */
  {
    int plane_size = nx * ny;

    fwrite(lattice_data, plane_size, sizeof(float), outfile);
    fwrite(lattice_data + (nz-1) * plane_size, plane_size, sizeof(float),
     outfile);
  }

  /* Cleanup */
  fclose(outfile);
}


int cpu_compute_cutoff_potential_lattice(
    Lattice *lattice,                  /* the lattice */
    float cutoff,                      /* cutoff distance */
    Atoms *atoms                       /* array of atoms */
    )
{
  int nx = lattice->dim.nx;
  int ny = lattice->dim.ny;
  int nz = lattice->dim.nz;
  float xlo = lattice->dim.lo.x;
  float ylo = lattice->dim.lo.y;
  float zlo = lattice->dim.lo.z;
  float gridspacing = lattice->dim.h;
  int natoms = atoms->size;
  Atom *atom = atoms->atoms;

  const float a2 = cutoff * cutoff;
  const float inv_a2 = 1.f / a2;
  float s;
  const float inv_gridspacing = 1.f / gridspacing;
  const int radius = (int) ceilf(cutoff * inv_gridspacing) - 1;
    /* lattice point radius about each atom */

  int n;
  int i, j, k;
  int ia, ib, ic;
  int ja, jb, jc;
  int ka, kb, kc;
  int index;
  int koff, jkoff;

  float x, y, z, q;
  float dx, dy, dz;
  float dz2, dydz2, r2;
  float e;
  float xstart, ystart;

  float *pg;

  int gindex;
  int ncell, nxcell, nycell, nzcell;
  int *first, *next;
  float inv_cellen = INV_CELLEN;
  Vec3 minext, maxext;    /* Extent of atom bounding box */
  //float xmin, ymin, zmin;
  //float xmax, ymax, zmax;

#if DEBUG_PASS_RATE
  unsigned long long pass_count = 0;
  unsigned long long fail_count = 0;
#endif

  /* find min and max extent */
  get_atom_extent(&minext, &maxext, atoms);

  /* number of cells in each dimension */
  nxcell = (int) floorf((maxext.x-minext.x) * inv_cellen) + 1;
  nycell = (int) floorf((maxext.y-minext.y) * inv_cellen) + 1;
  nzcell = (int) floorf((maxext.z-minext.z) * inv_cellen) + 1;
  ncell = nxcell * nycell * nzcell;

  /* allocate for cursor link list implementation */
  first = (int *) malloc(ncell * sizeof(int));
  for (gindex = 0;  gindex < ncell;  gindex++) {
    first[gindex] = -1;
  }
  next = (int *) malloc(natoms * sizeof(int));
  for (n = 0;  n < natoms;  n++) {
    next[n] = -1;
  }

  /* geometric hashing */
  for (n = 0;  n < natoms;  n++) {
    if (0==atom[n].q) continue;  /* skip any non-contributing atoms */
    i = (int) floorf((atom[n].x - minext.x) * inv_cellen);
    j = (int) floorf((atom[n].y - minext.y) * inv_cellen);
    k = (int) floorf((atom[n].z - minext.z) * inv_cellen);
    gindex = (k*nycell + j)*nxcell + i;
    next[n] = first[gindex];
    first[gindex] = n;
  }

  /* traverse the grid cells */
  for (gindex = 0;  gindex < ncell;  gindex++) {
    for (n = first[gindex];  n != -1;  n = next[n]) {
      x = atom[n].x - xlo;
      y = atom[n].y - ylo;
      z = atom[n].z - zlo;
      q = atom[n].q;

      /* find closest grid point with position less than or equal to atom */
      ic = (int) (x * inv_gridspacing);
      jc = (int) (y * inv_gridspacing);
      kc = (int) (z * inv_gridspacing);

      /* find extent of surrounding box of grid points */
      ia = ic - radius;
      ib = ic + radius + 1;
      ja = jc - radius;
      jb = jc + radius + 1;
      ka = kc - radius;
      kb = kc + radius + 1;

      /* trim box edges so that they are within grid point lattice */
      if (ia < 0)   ia = 0;
      if (ib >= nx) ib = nx-1;
      if (ja < 0)   ja = 0;
      if (jb >= ny) jb = ny-1;
      if (ka < 0)   ka = 0;
      if (kb >= nz) kb = nz-1;

      /* loop over surrounding grid points */
      xstart = ia*gridspacing - x;
      ystart = ja*gridspacing - y;
      dz = ka*gridspacing - z;
      for (k = ka;  k <= kb;  k++, dz += gridspacing) {
        koff = k*ny;
        dz2 = dz*dz;
        dy = ystart;
        for (j = ja;  j <= jb;  j++, dy += gridspacing) {
          jkoff = (koff + j)*nx;
          dydz2 = dy*dy + dz2;
#ifdef CHECK_CYLINDER_CPU
          if (dydz2 >= a2) continue;
#endif

          dx = xstart;
          index = jkoff + ia;
          pg = lattice->lattice + index;

#if defined(__INTEL_COMPILER)
          for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
            r2 = dx*dx + dydz2;
            s = (1.f - r2 * inv_a2) * (1.f - r2 * inv_a2);
            e = q * (1/sqrtf(r2)) * s;
            *pg += (r2 < a2 ? e : 0);  /* LOOP VECTORIZED!! */
          }
#else
          for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
            r2 = dx*dx + dydz2;
            if (r2 >= a2)
    {
#ifdef DEBUG_PASS_RATE
      fail_count++;
#endif
      continue;
    }
#ifdef DEBUG_PASS_RATE
      pass_count++;
#endif
            s = (1.f - r2 * inv_a2);
            e = q * (1/sqrtf(r2)) * s * s;
            *pg += e;
          }
#endif
        }
      } /* end loop over surrounding grid points */

    } /* end loop over atoms in a gridcell */
  } /* end loop over gridcells */

  /* free memory */
  free(next);
  free(first);

  /* For debugging: print the number of times that the test passed/failed */
#ifdef DEBUG_PASS_RATE
  printf ("Pass :%lld\n", pass_count);
  printf ("Fail :%lld\n", fail_count);
#endif

  return 0;
}
