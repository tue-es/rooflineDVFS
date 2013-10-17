
#include "parboil.h"
#include <stdio.h>
#include <stdlib.h>

#include "file.h"
#include "gpu_info.h"
#include "spmv_jds.h"
#include "jds_kernels.cu"
#include "convert_dataset.h"

#define ITERS 2

/*
static int generate_vector(float *x_vector, int dim) 
{	
	srand(54321);	
	for(int i=0;i<dim;i++)
	{
		x_vector[i] = (rand() / (float) RAND_MAX);
	}
	return 0;
}
*/

int main(int argc, char** argv) {
	//struct pb_TimerSet timers;
	//struct pb_Parameters *parameters;
	
	
	
	
	
	printf("CUDA accelerated sparse matrix vector multiplication****\n");
	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
	printf("This version maintained by Chris Rodrigues  ***********\n");
	//parameters = pb_ReadParameters(&argc, argv);
	
  /* Initialize the parameters structure */
  struct pb_Parameters *parameters = (struct pb_Parameters *)malloc(sizeof(struct pb_Parameters));
  parameters->outFile = NULL;
  parameters->inpFiles = (char **)malloc(sizeof(char *)*2);
  parameters->inpFiles[0] = NULL;
  
// Read input from command line
#ifdef SIZE0
  parameters->inpFiles[0] = "~/software/parboil-2.5/datasets/spmv/small/input/1138_bus.mtx";
  parameters->inpFiles[1] = "~/software/parboil-2.5/datasets/spmv/small/input/vector.bin";
#endif
#ifdef SIZE1
  parameters->inpFiles[0] = "~/software/parboil-2.5/datasets/spmv/medium/input/bcsstk18.mtx";
  parameters->inpFiles[1] = "~/software/parboil-2.5/datasets/spmv/medium/input/vector.bin";
#endif
  
	if ((parameters->inpFiles[0] == NULL) || (parameters->inpFiles[1] == NULL))
    {
      fprintf(stderr, "Expecting one two filenames\n");
      exit(-1);
    }

	
	//pb_InitializeTimerSet(&timers);
	//pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//parameters declaration
	int len;
	int depth;
	int dim;
	int pad=32;
	int nzcnt_len;
	
	//host memory allocation
	//matrix
	float *h_data;
	int *h_indices;
	int *h_ptr;
	int *h_perm;
	int *h_nzcnt;
	//vector
	float *h_Ax_vector;
  float *h_x_vector;
	
	//device memory allocation
	//matrix
	float *d_data;
	int *d_indices;
	int *d_ptr;
	int *d_perm;
	int *d_nzcnt;
	//vector
	float *d_Ax_vector;
    float *d_x_vector;
    

    //load matrix from files
	//pb_SwitchToTimer(&timers, pb_TimerID_IO);

	//inputData(parameters->inpFiles[0], &len, &depth, &dim,&nzcnt_len,&pad,
	//    &h_data, &h_indices, &h_ptr,
	//    &h_perm, &h_nzcnt);

	// HACK: remove the .bin from the end of data, remove later
	int col_count;
	//parameters->inpFiles[0][strlen(parameters->inpFiles[0])-4] = 0x00;
	printf("Input file %s\n", parameters->inpFiles[0]);
	 coo_to_jds(
		parameters->inpFiles[0], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
		1, // row padding
		pad, // warp size
		1, // pack size
		1, // is mirrored?
		0, // binary matrix
		1, // debug level [0:2]
		&h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
		&col_count, &dim, &len, &nzcnt_len, &depth
	);	

	int i;
	for (i=0; i<dim; i++) {
		//printf("%d = %d\n", h_perm[i], h_perm2[i]);
	}

  h_Ax_vector=(float*)malloc(sizeof(float)*dim);
  h_x_vector=(float*)malloc(sizeof(float)*dim);
  input_vec( parameters->inpFiles[1],h_x_vector,dim);
  
	//pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  
	
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
	
	
	//pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//memory allocation
	cudaMalloc((void **)&d_data, len*sizeof(float));
	cudaMalloc((void **)&d_indices, len*sizeof(int));
	cudaMalloc((void **)&d_ptr, depth*sizeof(int));
	cudaMalloc((void **)&d_perm, dim*sizeof(int));
	cudaMalloc((void **)&d_nzcnt, nzcnt_len*sizeof(int));
	cudaMalloc((void **)&d_x_vector, dim*sizeof(float));
	cudaMalloc((void **)&d_Ax_vector,dim*sizeof(float));
	cudaMemset( (void *) d_Ax_vector, 0, dim*sizeof(float));
	
	//memory copy
	cudaMemcpy(d_data, h_data, len*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, h_indices, len*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_perm, h_perm, dim*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_vector, h_x_vector, dim*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(jds_ptr_int, h_ptr, depth*sizeof(int));
	cudaMemcpyToSymbol(sh_zcnt_int, h_nzcnt,nzcnt_len*sizeof(int));
	
	cudaThreadSynchronize();
	//pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	unsigned int grid;
	unsigned int block;
    compute_active_thread(&block, &grid,nzcnt_len,pad, deviceProp.major,deviceProp.minor,
					deviceProp.warpSize,deviceProp.multiProcessorCount);
	
  //cudaFuncSetCacheConfig(spmv_jds_naive, cudaFuncCachePreferL1);

	//main execution
	//pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

	for (int i=0; i<ITERS; i++)
	spmv_jds_naive<<<grid, block>>>(d_Ax_vector,
  				d_data,d_indices,d_perm,
				d_x_vector,d_nzcnt,dim);
							
    CUERR // check and clear any existing errors
	
	cudaThreadSynchronize();
	
	//pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//HtoD memory copy
	cudaMemcpy(h_Ax_vector, d_Ax_vector,dim*sizeof(float), cudaMemcpyDeviceToHost);	

	cudaThreadSynchronize();
	cudaFree(d_data);
    cudaFree(d_indices);
    cudaFree(d_ptr);
	cudaFree(d_perm);
    cudaFree(d_nzcnt);
    cudaFree(d_x_vector);
	cudaFree(d_Ax_vector);
 
	if (parameters->outFile) {
		//pb_SwitchToTimer(&timers, pb_TimerID_IO);
		//int temp = ((dim + 31)/32)*32; // hack because of "gold" version including padding
		outputData(parameters->outFile,h_Ax_vector,dim);
		
	}
	//pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	free (h_data);
	free (h_indices);
	free (h_ptr);
	free (h_perm);
	free (h_nzcnt);
	free (h_Ax_vector);
	free (h_x_vector);
	//pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	//pb_PrintTimerSet(&timers);
	//pb_FreeParameters(parameters);

	return 0;

}



void inputData(char* fName, int* len, int* depth, int* dim,int *nzcnt_len,int *pad,
               float** h_data, int** h_indices, int** h_ptr,
               int** h_perm, int** h_nzcnt)
{
  FILE* fid = fopen(fName, "rb");

  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open input file\n");
      exit(-1);
    }

  size_t temp;
  temp = fscanf(fid, "%d %d %d %d %d\n",len,depth,nzcnt_len,dim,pad);
  int _len=len[0];
  int _depth=depth[0];
  int _dim=dim[0];
  //int _pad=pad[0];
  int _nzcnt_len=nzcnt_len[0];
  
  *h_data = (float *) malloc(_len * sizeof (float));
  temp = fread (*h_data, sizeof (float), _len, fid);
  
  *h_indices = (int *) malloc(_len * sizeof (int));
  temp = fread (*h_indices, sizeof (int), _len, fid);
  
  *h_ptr = (int *) malloc(_depth * sizeof (int));
  temp = fread (*h_ptr, sizeof (int), _depth, fid);
  
  *h_perm = (int *) malloc(_dim * sizeof (int));
  temp = fread (*h_perm, sizeof (int), _dim, fid);
  
  *h_nzcnt = (int *) malloc(_nzcnt_len * sizeof (int));
  temp = fread (*h_nzcnt, sizeof (int), _nzcnt_len, fid);
  temp += _len;
  fclose (fid); 
}

void input_vec(char *fName,float *h_vec,int dim)
{
  FILE* fid = fopen(fName, "rb");
  size_t temp = fread (h_vec, sizeof (float), dim, fid);
  temp += dim;
  fclose(fid);
  
}

void outputData(char* fName, float *h_Ax_vector,int dim)
{
  FILE* fid = fopen(fName, "w");
  uint32_t tmp32;
  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open output file\n");
      exit(-1);
    }
  tmp32 = dim;
  fwrite(&tmp32, sizeof(uint32_t), 1, fid);
  fwrite(h_Ax_vector, sizeof(float), dim, fid);

  fclose (fid);
}


void compute_active_thread(unsigned int *thread,
					unsigned int *grid,
					int task,
					int pad,
					int major,
					int minor,
					int warp_size,
					int sm)
{
	int max_thread;
	//int max_warp;
	int max_block=8;
	if(major==1)
	{
		if(minor>=2)
		{
			max_thread=1024;
			//max_warp=32;
		}
		else
		{
			max_thread=768;
			//max_warp=24;
		}
	}
	else if(major==2)
	{
		max_thread=1536;
		//max_warp=48;
	}
	else
	{
		//newer GPU  //keep using 2.0
		max_thread=1536;
		//max_warp=48;
	}
	
	int _grid;
	int _thread;
	//int threads_per_sm=0;
	if(task*pad>sm*max_thread)
	{
		//_grid=sm*max_block;
		_thread=max_thread/max_block;
		_grid=(task*pad+_thread-1)/_thread;
	}
	else
	{
		_thread=pad;
		_grid=task;
	}
	thread[0]=_thread;
	grid[0]=_grid;
	
}

/*
*   NOTES:
*
*   1) Matrix Market files are always 1-based, i.e. the index of the first
*      element of a matrix is (1,1), not (0,0) as in C.  ADJUST THESE
*      OFFSETS ACCORDINGLY when reading and writing 
*      to files.
*
*   2) ANSI C requires one to use the "l" format modifier when reading
*      double precision floating point numbers in scanf() and
*      its variants.  For example, use "%lf", "%lg", or "%le"
*      when reading doubles, otherwise errors will occur.
*/

#include "convert_dataset.h"

typedef struct _mat_entry {
    int row, col; /* i,j */
    float val;
} mat_entry;

typedef struct _row_stats { // stats on each row
    int index;
    int size;
    int start;
    int padding;
} row_stats;

int sort_rows(const void* a, const void* b) {
    return (((mat_entry*)a)->row - ((mat_entry*)b)->row);
}
int sort_cols(const void* a, const void* b) {
    return (((mat_entry*)a)->col - ((mat_entry*)b)->col);
}
/* sorts largest by size first */
int sort_stats(const void* a, const void* b) {
    return(((row_stats*)b)->size - ((row_stats*)a)->size);
}

/*
 * COO to JDS matrix conversion.
 *
 * Needs to output both column and row major JDS formats
 * with the minor unit padded to a multiple of `pad_minor`
 * and the major unit arranged into groups of `group_size`
 *
 * Major unit is col, minor is row. Each block is either a scalar or vec4
 *
 * Inputs:
 *   mtx_filename - the file in COO format
 *   pad_rows - multiple of packed groups to pad each row to
 *   warp_size - each group of `warp_size` cols is padded to the same amount
 *   pack_size - number of items to pack
 *   mirrored - is the input mtx file a symmetric matrix? The other half will be
 *   	filled in if this is =1
 *   binary - does the sparse matrix file have values in the format "%d %d"
 *   	or "%d %d %lg"?
 *   debug_level - 0 for no output, 1 for simple JDS data, 2 for visual grid
 * Outputs:
 *   data - the raw data, padded and grouped as requested
 *   data_row_ptr - pointer offset into the `data` output, referenced
 *      by the current row loop index
 *   nz_count - number of non-zero entries in each row
 *      indexed by col / warp_size
 *   data_col_index - corresponds to the col that the same
 *      array index in `data` is at
 *   data_row_map - JDS row to real row
 *   data_cols - number of columns the output JDS matrix has
 *   dim - dimensions of the input matrix
 *   data_ptr_len - size of data_row_ptr (maps to original `depth` var)
 */
int coo_to_jds(char* mtx_filename, int pad_rows, int warp_size, int pack_size,
	       int mirrored, int binary, int debug_level,
               float** data, int** data_row_ptr, int** nz_count, int** data_col_index,
               int** data_row_map, int* data_cols, int* dim, int* len, int* nz_count_len,
	       int* data_ptr_len) {
    //int ret_code;
    MM_typecode matcode;
    FILE *f;
    int nz;   
    int i;
    //float *val;
    mat_entry* entries;
    row_stats* stats;
    int rows, cols;
    size_t temp;
    
    if ((f = fopen(mtx_filename, "r")) == NULL) 
        exit(1);


    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((mm_read_mtx_crd_size(f, &rows, &cols, &nz)) !=0)
        exit(1);
    *dim = rows;
    
    if (mirrored) {
	// max possible size, might be less because diagonal values aren't doubled
	entries = (mat_entry*) malloc(2 * nz * sizeof(mat_entry));
    } else {
	entries = (mat_entry*) malloc(nz * sizeof(mat_entry));
    }
    
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    int cur_i=0; // to account for mirrored diagonal entries

    for (i=0; i<nz; i++, cur_i++)
    {
	if (!binary) {
	    temp = fscanf(f, "%d %d %f\n", &entries[cur_i].row, &entries[cur_i].col, &entries[cur_i].val);
	} else {
	    temp = fscanf(f, "%d %d\n", &entries[cur_i].row, &entries[cur_i].col);
	    entries[cur_i].val = 1.0;
	}
        entries[cur_i].row--;
        entries[cur_i].col--;
	//printf("%d,%d = %f\n", entries[cur_i].row, entries[cur_i].col, entries[cur_i].val);
	if (mirrored) {
	    // fill in mirrored diagonal
	    if (entries[cur_i].row != entries[cur_i].col) { // not a diagonal value
		cur_i++;
		entries[cur_i].val = entries[cur_i-1].val;
		entries[cur_i].col = entries[cur_i-1].row;
		entries[cur_i].row = entries[cur_i-1].col;
		//printf("%d,%d = %f\n", entries[cur_i].row, entries[cur_i].col, entries[cur_i].val);
	    }
	}
    }
    // set new non-zero count
    nz = cur_i;
    if (debug_level >= 1) {
	printf("Converting COO to JDS format (%dx%d)\n%d matrix entries, warp size = %d, "
	       "row padding align = %d, pack size = %d\n\n", rows, cols, nz, warp_size, pad_rows, pack_size);
    }
    if (f !=stdin) fclose(f);

    /*
     * Now we have an array of values in entries
     * Transform to padded JDS format  - sort by rows, then fubini
     */

    int irow, icol=0, istart=0;
    int total_size=0;

    /* Loop through each entry to figure out padding, grouping that determine
     * final data array size
     *
     * First calculate stats for each row
     * 
     * Collect stats using the major_stats typedef
     */
    
    
    qsort(entries, nz, sizeof(mat_entry), sort_rows); // sort by row number
    rows = entries[nz-1].row+1; // last item is greatest row (zero indexed)
    if (rows%warp_size) { // pad group number to warp_size here
	rows += warp_size - rows%warp_size;
    }
    stats = (row_stats*) calloc(rows, sizeof(row_stats)); // set to 0
    *data_row_map = (int*) calloc(rows, sizeof(int));
    irow = entries[0].row; // set first row
    
    //printf("First row %d\n", irow);
    for (i=0; i<nz; i++) { // loop through each sorted entry
	if (entries[i].row != irow || i == nz-1) { // new row
	    //printf("%d != %d\n", entries[i].row, irow);
	    if (i == nz-1) {
		// last item, add it to current row
		//printf("Last item i=%d, row=%d, irow=%d\n", i, entries[i].row, irow);
		icol++;
	    }
	    // hit a new row, record stats for the last row (i-1)
	    stats[irow].size = icol; // record # cols in previous row
	    stats[irow].index = entries[i-1].row; // row # for previous stat item
	    //printf("Row %d, i=%d, irow=%d\n", entries[i].row, i, irow);
	    stats[irow].start = istart; // starting location in entries array
	    // set stats for the next row until this break again
	    icol=0; // reset row items
	    irow = entries[i].row;
	    istart = i;
	}
	icol++; // keep track of number of items in this row
    }
    
    
    *nz_count_len = rows/warp_size + rows%warp_size;
    *nz_count = (int*) malloc(*nz_count_len * sizeof(int)); // only one value per group
    
    /* sort based upon row size, greatest first */
    qsort(stats, rows, sizeof(row_stats), sort_stats);
    /* figure out padding and grouping */
    if (debug_level >= 1) {
	printf("Padding data....%d rows, %d groups\n", rows, *nz_count_len);
    }
    int pad_to = 0, total_padding = 0, pack_to;
    pad_rows *= pack_size; // change padding to account for packed items
    for (i=0; i<rows; i++) {
	// record JDS to real row number
	(*data_row_map)[i] = stats[i].index;
	if (i<rows-1) {
	   // (*data_row_map)[i]--; // ???? no idea why this is off by 1
	}
	// each row is padded so the number of packed groups % pad_rows == 0
	if (i % warp_size == 0) { // on a group boundary with the largest number of items
	    // find padding in individual items
	    if (stats[i].size % pad_rows) {
		stats[i].padding = pad_rows - (stats[i].size % pad_rows); // find padding
	    } else {
		stats[i].padding = 0; // no padding necessary, already at pad multiple
	    }
	    if (stats[i].size % pack_size) {
		pack_to = ceil((float)stats[i].size/pack_size);
	    } else {
		pack_to = stats[i].size/pack_size;
	    }
	    //pack_to = stats[i].size + (!stats[i].size%pack_size) ? 0 : (pack_size - stats[i].size%pack_size);
	    pad_to = stats[i].size + stats[i].padding; // total size of this row, with padding
	    // TODO: change this to reflect the real number of nonzero packed items, not the padded
	    // value
	    (*nz_count)[i/warp_size] = pack_to; // number of packed items in this group
	    total_size += pad_to * warp_size; // allocate size for this padded group
	    if (debug_level >= 2)
		printf("Padding warp group %d to %d items, zn = %d\n", i/warp_size, pad_to, pack_to);
	} else {
	    stats[i].padding = pad_to - stats[i].size;
	}
	total_padding += stats[i].padding;
	//if (debug_level >= 2)
	//    printf("Row %d, %d items, %d padding\n", stats[i].index, stats[i].size, stats[i].padding);
    }
    
    /* allocate data and data_row_index */
    if (debug_level >= 1)
	printf("Allocating data space: %d entries (%f%% padding)\n", total_size, (float)100*total_padding/total_size);
    *data = (float*) calloc(total_size, sizeof(float)); // set to 0 so padded values are set
    *data_col_index = (int*) calloc(total_size, sizeof(int)); // any unset indexes point to 0
    *data_row_ptr = (int*) calloc(rows, sizeof(int));
    *len = total_size;
    i = 0; // data index, including padding
    
    /*
     * Keep looping through each row, writing data a group at a time
     * to the output array. Increment `irow` each time, and use it as
     * an index into entries along with stats.start to get the next
     * data item
     */
    irow = 0; // keep track of which row we are in inside the fubini-ed array
    int idata = 0; // position within final data array
    int entry_index;
    int ipack; // used in internal loop for writing packed values
    mat_entry entry;
    while (1) {
	/* record data_row_ptr */
	(*data_row_ptr)[irow] = idata;
	
	/* End condtion: the size of the greatest row is smaller than the current
	  Fubini-ed row */
	if (stats[0].size+stats[0].padding <= irow*pack_size) break;

	//printf("Data row pointer for row %d is %d\n", irow, idata);
	for (i=0; i<rows; i++) {
	    /* take one packed group from each original row */
	    //printf("Output irow %d icol %d (real %d,%d size %d)\n", irow, i, entry.col, i, stats[i].size);
	    /* Watch out for little vs big endian, and how opencl interprets vector casting from pointers */
	    for (ipack=0; ipack<pack_size; ipack++) {
		if (stats[i].size > irow*pack_size+ipack) {
		    // copy value
		    entry_index = stats[i].start + irow*pack_size+ipack;
		    entry = entries[entry_index];
		    /* record index and value */
		    (*data)[idata] = entry.val;
		    /* each data item will get its row index from the thread, col from here */
		    (*data_col_index)[idata] = entry.col;

		    if (debug_level >= 2) {
			if (i < 3) {
			    // first row debugging
			    printf("[%d row%d=%.3f]", ipack+1, i, entry.val);
			} else {
			    printf("%d", ipack+1);
			}
		    }
		} else if (stats[i].size+stats[i].padding > irow*pack_size+ipack) {
		    /* add padding to the end of each row here - this assumes padding is factored into allocated size */
		    if (debug_level >= 2) printf("0");
		    (*data_col_index)[idata] = -1;
		} else {
		    goto endwrite; // no data written this pass, so don't increment idata
		}
		idata += 1;
	    }
	}
	endwrite:
	if (debug_level >= 2) {
	    printf("\n");
	}
	irow += 1;
    }
    
    if (debug_level >= 1)
	printf("Finished converting.\nJDS format has %d columns, %d rows.\n", rows, irow);
    free(entries);
    free(stats);
    printf("nz_count_len = %d\n", *nz_count_len);
    
    *data_cols = rows;
    *data_ptr_len = irow+1;
    temp += irow;
    return 0;
}


#include "mmio.h"

int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,
                double **val_, int **I_, int **J_)
{
    FILE *f;
    MM_typecode matcode;
    int M, N, nz;
    int i;
    double *val;
    int *I, *J;
    size_t temp;
 
    if ((f = fopen(fname, "r")) == NULL)
            return -1;
 
 
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return -1;
    }
 
 
 
    if ( !(mm_is_real(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode)))
    {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n",
                mm_typecode_to_str(matcode));
        return -1;
    }
 
    /* find out size of sparse matrix: M, N, nz .... */
 
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {
        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");
        return -1;
    }
 
    *M_ = M;
    *N_ = N;
    *nz_ = nz;
 
    /* reseve memory for matrices */
 
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));
 
    *val_ = val;
    *I_ = I;
    *J_ = J;
 
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
 
    for (i=0; i<nz; i++)
    {
        temp = fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    fclose(f);
    i += temp;
    return 0;
}

int mm_is_valid(MM_typecode matcode)
{
    if (!mm_is_matrix(matcode)) return 0;
    if (mm_is_dense(matcode) && mm_is_pattern(matcode)) return 0;
    if (mm_is_real(matcode) && mm_is_hermitian(matcode)) return 0;
    if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) || 
                mm_is_skew(matcode))) return 0;
    return 1;
}

int mm_read_banner(FILE *f, MM_typecode *matcode)
{
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH]; 
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;


    mm_clear_typecode(matcode);  

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL) 
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, 
        storage_scheme) != 5)
        return MM_PREMATURE_EOF;

    for (p=mtx; *p!='\0'; *p=tolower(*p),p++);  /* convert to lower case */
    for (p=crd; *p!='\0'; *p=tolower(*p),p++);  
    for (p=data_type; *p!='\0'; *p=tolower(*p),p++);
    for (p=storage_scheme; *p!='\0'; *p=tolower(*p),p++);

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0)
        return  MM_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);


    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */


    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matcode);
    else
    if (strcmp(crd, MM_DENSE_STR) == 0)
            mm_set_dense(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
    

    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else
    if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matcode);
    else
    if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matcode);
    else
    if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
    

    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else
    if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matcode);
    else
    if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matcode);
    else
    if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
        

    return 0;
}

int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz)
{
    if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)
        return MM_COULD_NOT_WRITE_FILE;
    else 
        return 0;
}

int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz )
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    /* set return null parameter values, in case we exit with errors */
    *M = *N = *nz = 0;

    /* now continue scanning until you reach the end-of-comments */
    do 
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL) 
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
        return 0;
        
    else
    do
    { 
        num_items_read = fscanf(f, "%d %d %d", M, N, nz); 
        if (num_items_read == EOF) return MM_PREMATURE_EOF;
    }
    while (num_items_read != 3);

    return 0;
}


int mm_read_mtx_array_size(FILE *f, int *M, int *N)
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;
    /* set return null parameter values, in case we exit with errors */
    *M = *N = 0;
	
    /* now continue scanning until you reach the end-of-comments */
    do 
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL) 
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d", M, N) == 2)
        return 0;
        
    else /* we have a blank line */
    do
    { 
        num_items_read = fscanf(f, "%d %d", M, N); 
        if (num_items_read == EOF) return MM_PREMATURE_EOF;
    }
    while (num_items_read != 2);

    return 0;
}

int mm_write_mtx_array_size(FILE *f, int M, int N)
{
    if (fprintf(f, "%d %d\n", M, N) != 2)
        return MM_COULD_NOT_WRITE_FILE;
    else 
        return 0;
}



/*-------------------------------------------------------------------------*/

/******************************************************************/
/* use when I[], J[], and val[]J, and val[] are already allocated */
/******************************************************************/

int mm_read_mtx_crd_data(FILE *f, int M, int N, int nz, int I[], int J[],
        double val[], MM_typecode matcode)
{
    int i;
    if (mm_is_complex(matcode))
    {
        for (i=0; i<nz; i++)
            if (fscanf(f, "%d %d %lg %lg", &I[i], &J[i], &val[2*i], &val[2*i+1])
                != 4) return MM_PREMATURE_EOF;
    }
    else if (mm_is_real(matcode))
    {
        for (i=0; i<nz; i++)
        {
            if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i])
                != 3) return MM_PREMATURE_EOF;

        }
    }

    else if (mm_is_pattern(matcode))
    {
        for (i=0; i<nz; i++)
            if (fscanf(f, "%d %d", &I[i], &J[i])
                != 2) return MM_PREMATURE_EOF;
    }
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;
        
}

int mm_read_mtx_crd_entry(FILE *f, int *I, int *J,
        double *real, double *imag, MM_typecode matcode)
{
    if (mm_is_complex(matcode))
    {
            if (fscanf(f, "%d %d %lg %lg", I, J, real, imag)
                != 4) return MM_PREMATURE_EOF;
    }
    else if (mm_is_real(matcode))
    {
            if (fscanf(f, "%d %d %lg\n", I, J, real)
                != 3) return MM_PREMATURE_EOF;

    }

    else if (mm_is_pattern(matcode))
    {
            if (fscanf(f, "%d %d", I, J) != 2) return MM_PREMATURE_EOF;
    }
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;
        
}


/************************************************************************
    mm_read_mtx_crd()  fills M, N, nz, array of values, and return
                        type code, e.g. 'MCRS'

                        if matrix is complex, values[] is of size 2*nz,
                            (nz pairs of real/imaginary values)
************************************************************************/

int mm_read_mtx_crd(char *fname, int *M, int *N, int *nz, int **I, int **J, 
        double **val, MM_typecode *matcode)
{
    int ret_code;
    FILE *f;

    if (strcmp(fname, "stdin") == 0) f=stdin;
    else
    if ((f = fopen(fname, "r")) == NULL)
        return MM_COULD_NOT_READ_FILE;


    if ((ret_code = mm_read_banner(f, matcode)) != 0)
        return ret_code;

    if (!(mm_is_valid(*matcode) && mm_is_sparse(*matcode) && 
            mm_is_matrix(*matcode)))
        return MM_UNSUPPORTED_TYPE;

    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)
        return ret_code;


    *I = (int *)  malloc(*nz * sizeof(int));
    *J = (int *)  malloc(*nz * sizeof(int));
    *val = NULL;

    if (mm_is_complex(*matcode))
    {
        *val = (double *) malloc(*nz * 2 * sizeof(double));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) return ret_code;
    }
    else if (mm_is_real(*matcode))
    {
        *val = (double *) malloc(*nz * sizeof(double));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) return ret_code;
    }

    else if (mm_is_pattern(*matcode))
    {
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) return ret_code;
    }

    if (f != stdin) fclose(f);
    return 0;
}

int mm_write_banner(FILE *f, MM_typecode matcode)
{
    char *str = mm_typecode_to_str(matcode);
    int ret_code;

    ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, str);
    free(str);
    if (ret_code !=2 )
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[],
        double val[], MM_typecode matcode)
{
    FILE *f;
    int i;

    if (strcmp(fname, "stdout") == 0) 
        f = stdout;
    else
    if ((f = fopen(fname, "w")) == NULL)
        return MM_COULD_NOT_WRITE_FILE;
    
    /* print banner followed by typecode */
    fprintf(f, "%s ", MatrixMarketBanner);
    fprintf(f, "%s\n", mm_typecode_to_str(matcode));

    /* print matrix sizes and nonzeros */
    fprintf(f, "%d %d %d\n", M, N, nz);

    /* print values */
    if (mm_is_pattern(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d\n", I[i], J[i]);
    else
    if (mm_is_real(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d %20.16g\n", I[i], J[i], val[i]);
    else
    if (mm_is_complex(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d %20.16g %20.16g\n", I[i], J[i], val[2*i], 
                        val[2*i+1]);
    else
    {
        if (f != stdout) fclose(f);
        return MM_UNSUPPORTED_TYPE;
    }

    if (f !=stdout) fclose(f);

    return 0;
}
  

/**
*  Create a new copy of a string s.  mm_strdup() is a common routine, but
*  not part of ANSI C, so it is included here.  Used by mm_typecode_to_str().
*
*/
char *mm_strdup(const char *s)
{
	int len = strlen(s);
	char *s2 = (char *) malloc((len+1)*sizeof(char));
	return strcpy(s2, s);
}

char  *mm_typecode_to_str(MM_typecode matcode)
{
    char buffer[MM_MAX_LINE_LENGTH];
    char *types[4];
	char *mm_strdup(const char *);
    //int error =0;

    /* check for MTX type */
    if (mm_is_matrix(matcode)) 
        types[0] = MM_MTX_STR;
    //else
    //    error=1;

    /* check for CRD or ARR matrix */
    if (mm_is_sparse(matcode))
        types[1] = MM_SPARSE_STR;
    else
    if (mm_is_dense(matcode))
        types[1] = MM_DENSE_STR;
    else
        return NULL;

    /* check for element data type */
    if (mm_is_real(matcode))
        types[2] = MM_REAL_STR;
    else
    if (mm_is_complex(matcode))
        types[2] = MM_COMPLEX_STR;
    else
    if (mm_is_pattern(matcode))
        types[2] = MM_PATTERN_STR;
    else
    if (mm_is_integer(matcode))
        types[2] = MM_INT_STR;
    else
        return NULL;


    /* check for symmetry type */
    if (mm_is_general(matcode))
        types[3] = MM_GENERAL_STR;
    else
    if (mm_is_symmetric(matcode))
        types[3] = MM_SYMM_STR;
    else 
    if (mm_is_hermitian(matcode))
        types[3] = MM_HERM_STR;
    else 
    if (mm_is_skew(matcode))
        types[3] = MM_SKEW_STR;
    else
        return NULL;

    sprintf(buffer,"%s %s %s %s", types[0], types[1], types[2], types[3]);
    return mm_strdup(buffer);

}
