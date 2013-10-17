/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/


#include "parboil.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include "util.h"
#include "bmp.h"

#include "histo_final.cu"
#include "histo_intermediates.cu"
#include "histo_main.cu"
#include "histo_prescan.cu"
 
#define ITERS 2

/******************************************************************************
* Implementation: GPU
* Details:
* in the GPU implementation of histogram, we begin by computing the span of the
* input values into the histogram. Then the histogramming computation is carried
* out by a (BLOCK_X, BLOCK_Y) sized grid, where every group of Y (same X)
* computes its own partial histogram for a part of the input, and every Y in the
* group exclusively writes to a portion of the span computed in the beginning.
* Finally, a reduction is performed to combine all the partial histograms into
* the final result.
******************************************************************************/

int main(int argc, char* argv[]) {
  //struct pb_TimerSet *timersPtr;
  //struct pb_Parameters *parameters;

  /* Initialize the parameters structure */
  struct pb_Parameters *parameters = (struct pb_Parameters *)malloc(sizeof(struct pb_Parameters));
  parameters->outFile = NULL;
  parameters->inpFiles = (char **)malloc(sizeof(char *));
  parameters->inpFiles[0] = NULL;
  
// Read input from command line
#ifdef SIZE0 // 20 4
  parameters->inpFiles[0] = "~/software/parboil-2.5/datasets/histo/default/input/img.bin";
#endif
#ifdef SIZE1 // 10000 4
  parameters->inpFiles[0] = "~/software/parboil-2.5/datasets/histo/large/input/img.bin";
#endif
  
  //parameters = pb_ReadParameters(&argc, argv);
  if (!parameters)
    return -1;

  if(!parameters->inpFiles[0]){
    fputs("Input file expected\n", stderr);
    return -1;
  }

  //timersPtr = (struct pb_TimerSet *) malloc (sizeof(struct pb_TimerSet));
  
  
  //appendDefaultTimerSet(NULL);
  
  
  //if (timersPtr == NULL) {
  //  fprintf(stderr, "Could not append default timer set!\n");
  //  exit(1);
  //}
  
  //struct pb_TimerSet timers = *timersPtr;
  
//  pb_CreateTimer(&timers, "myTimer!", 0);
  
  
  //pb_InitializeTimerSet(&timers);
  
  //pb_AddSubTimer(&timers, "Input", pb_TimerID_IO);
  //pb_AddSubTimer(&timers, "Output", pb_TimerID_IO);
  /*
  char *prescans = "PreScanKernel";
  char *postpremems = "PostPreMems";
  char *intermediates = "IntermediatesKernel";
  char *mains = "MainKernel";
  char *finals = "FinalKernel";
  */
  //pb_AddSubTimer(&timers, prescans, pb_TimerID_KERNEL);
  //pb_AddSubTimer(&timers, postpremems, pb_TimerID_KERNEL);
  //pb_AddSubTimer(&timers, intermediates, pb_TimerID_KERNEL);
  //pb_AddSubTimer(&timers, mains, pb_TimerID_KERNEL);
  //pb_AddSubTimer(&timers, finals, pb_TimerID_KERNEL);
  
//  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  //pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

  int numIterations = ITERS;
  /*
  if (argc >= 2){
    numIterations = atoi(argv[1]);
  } else {
    fputs("Expected at least one command line argument\n", stderr);
    return -1;
  }*/

  unsigned int img_width, img_height;
  unsigned int histo_width, histo_height;

  FILE* f = fopen(parameters->inpFiles[0],"rb");
  int result = 0;

  result += fread(&img_width,    sizeof(unsigned int), 1, f);
  result += fread(&img_height,   sizeof(unsigned int), 1, f);
  result += fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += fread(&histo_height, sizeof(unsigned int), 1, f);

  if (result != 4){
    fputs("Error reading input and output dimensions from file\n", stderr);
    return -1;
  }

  unsigned int* img = (unsigned int*) malloc (img_width*img_height*sizeof(unsigned int));
  unsigned char* histo = (unsigned char*) calloc (histo_width*histo_height, sizeof(unsigned char));

  result = fread(img, sizeof(unsigned int), img_width*img_height, f);

  fclose(f);

  if (result != img_width*img_height){
    fputs("Error reading input array from file\n", stderr);
    return -1;
  }

  int even_width = ((img_width+1)/2)*2;
  unsigned int* input;
  unsigned int* ranges;
  uchar4* sm_mappings;
  unsigned int* global_subhisto;
  unsigned short* global_histo;
  unsigned int* global_overflow;
  unsigned char* final_histo;

  cudaMalloc((void**)&input           , even_width*(((img_height+UNROLL-1)/UNROLL)*UNROLL)*sizeof(unsigned int));
  cudaMalloc((void**)&ranges          , 2*sizeof(unsigned int));
  cudaMalloc((void**)&sm_mappings     , img_width*img_height*sizeof(uchar4));
  cudaMalloc((void**)&global_subhisto , img_width*histo_height*sizeof(unsigned int));
  cudaMalloc((void**)&global_histo    , img_width*histo_height*sizeof(unsigned short));
  cudaMalloc((void**)&global_overflow , img_width*histo_height*sizeof(unsigned int));
  cudaMalloc((void**)&final_histo     , img_width*histo_height*sizeof(unsigned char));

  cudaMemset(final_histo           ,0 , img_width*histo_height*sizeof(unsigned char));

  for (unsigned y=0; y < img_height; y++){
    cudaMemcpy(&(((unsigned int*)input)[y*even_width]),&img[y*img_width],img_width*sizeof(unsigned int), cudaMemcpyHostToDevice);
  }
  
  //pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
  //pb_SwitchToSubTimer(&timers, NULL, pb_TimerID_KERNEL);
  
  
  unsigned int *zeroData = (unsigned int *) calloc(img_width*histo_height, sizeof(unsigned int));
  

  for (int iter = 0; iter < numIterations; iter++) {
    unsigned int ranges_h[2] = {UINT32_MAX, 0};

    cudaMemcpy(ranges,ranges_h, 2*sizeof(unsigned int), cudaMemcpyHostToDevice);


    //pb_SwitchToSubTimer(&timers, prescans , pb_TimerID_KERNEL);

    histo_prescan_kernel<<<dim3(PRESCAN_BLOCKS_X),dim3(PRESCAN_THREADS)>>>((unsigned int*)input, img_height*img_width, ranges);

    //pb_SwitchToSubTimer(&timers, postpremems , pb_TimerID_KERNEL);

    cudaMemcpy(ranges_h,ranges, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaMemcpy(global_subhisto,zeroData, img_width*histo_height*sizeof(unsigned int), cudaMemcpyHostToDevice);
    //    cudaMemset(global_subhisto,0,img_width*histo_height*sizeof(unsigned int));

    //pb_SwitchToSubTimer(&timers, intermediates, pb_TimerID_KERNEL);
    histo_intermediates_kernel<<<dim3((img_height + UNROLL-1)/UNROLL), dim3((img_width+1)/2)>>>(
                (uint2*)(input),
                (unsigned int)img_height,
                (unsigned int)img_width,
                (img_width+1)/2,
                (uchar4*)(sm_mappings)
    );
    //pb_SwitchToSubTimer(&timers, mains, pb_TimerID_KERNEL);    
    
    histo_main_kernel<<<dim3(BLOCK_X, ranges_h[1]-ranges_h[0]+1), dim3(THREADS)>>>(
                (uchar4*)(sm_mappings),
                img_height*img_width,
                ranges_h[0], ranges_h[1],
                histo_height, histo_width,
                (unsigned int*)(global_subhisto),
                (unsigned int*)(global_histo),
                (unsigned int*)(global_overflow)
    );
    
    //pb_SwitchToSubTimer(&timers, finals, pb_TimerID_KERNEL);    

    histo_final_kernel<<<dim3(BLOCK_X*3), dim3(512)>>>(
                ranges_h[0], ranges_h[1],
                histo_height, histo_width,
                (unsigned int*)(global_subhisto),
                (unsigned int*)(global_histo),
                (unsigned int*)(global_overflow),
                (unsigned int*)(final_histo)
    );
  }

  //pb_SwitchToSubTimer(&timers, "Output", pb_TimerID_IO);
  //  pb_SwitchToTimer(&timers, pb_TimerID_IO);


  cudaMemcpy(histo,final_histo, histo_height*histo_width*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(input);
  cudaFree(ranges);
  cudaFree(sm_mappings);
  cudaFree(global_subhisto);
  cudaFree(global_histo);
  cudaFree(global_overflow);
  cudaFree(final_histo);

  if (parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  //pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  //pb_SwitchToSubTimer(&timers, NULL, pb_TimerID_COMPUTE);

  free(img);
  free(histo);

  //pb_SwitchToSubTimer(&timers, NULL, pb_TimerID_NONE);
  
  printf("\n");
  //pb_PrintTimerSet(&timers);
  //pb_FreeParameters(parameters);

  return 0;
}



// This function takes an HSV value and converts it to BMP.
// We use this function to generate colored images with
// Smooth spectrum traversal for the input and output images.
RGB HSVtoRGB( float h, float s, float v )
{
    int i;
    float f, p, q, t;
    float r, g, b;
    RGB value={0,0,0};

    if( s == 0 ) {
        r = g = b = v;
        return value;
    }
    h /= 60;
    i = floor( h );
    f = h - i;
    p = v * ( 1 - s );
    q = v * ( 1 - s * f );
    t = v * ( 1 - s * ( 1 - f ) );
    switch( i ) {
        case 0:
            r = v; g = t; b = p;
            break;
        case 1:
            r = q; g = v; b = p;
            break;
        case 2:
            r = p; g = v; b = t;
            break;
        case 3:
            r = p; g = q; b = v;
            break;
        case 4:
            r = t; g = p; b = v;
            break;
        default:
            r = v; g = p; b = q;
            break;
    }

    unsigned int temp = r*255;
    value.R = temp;
    temp = g*255;
    value.G = temp;
    temp = b*255;
    value.B = temp;

    return value;
}

void dump_histo_img(unsigned char* histo, unsigned int height, unsigned int width, const char *filename)
{
    RGB* pixel_map = (RGB*) malloc (height*width*sizeof(RGB));

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            unsigned char value = histo[y * width + x];

            if (value == 0){
                pixel_map[y*width+x].R = 0;
                pixel_map[y*width+x].G = 0;
                pixel_map[y*width+x].B = 0;
            } else {
                pixel_map[y*width+x] = HSVtoRGB(0.0,1.0,cbrt(1+ 63.0*((float)value)/((float)UINT8_MAX))/4);
            }
        }
    }
    create_bmp(pixel_map, height, width, filename);
    free(pixel_map);
}
