
// Includes
#include <stdio.h>

// Constants
#define WPT 128
#define THREADS 512
#define BLOCKS 14*2*4
#define N (BLOCKS*THREADS)

// Kernel
__global__ void pebench(unsigned *A, unsigned *B) {
	unsigned i = blockIdx.x*THREADS + threadIdx.x;
	unsigned acc = A[i];
	for (unsigned w=0; w<WPT; w++) {
		acc = acc * acc;
	}
	B[i] = acc;
}

// Timers
cudaEvent_t start;
void timer_start();
void timer_stop();

// Main function
int main(void) {
	unsigned size = N*sizeof(unsigned);
	
	// Allocate and initialise the data
	unsigned *A = (unsigned *)malloc(size);
	unsigned *B = (unsigned *)malloc(size);
	for (unsigned i=0; i<N; i++) {
		A[i] = i;
		B[i] = 0;
	}
	
	// Allocate CUDA arrays
	unsigned *devA = 0;
	unsigned *devB = 0;
	cudaMalloc(&devA, size);
	cudaMalloc(&devB, size);
	
	// Copy to the GPU
	cudaMemcpy(devA, A, size, cudaMemcpyHostToDevice);
	
	// Configure the kernel
	dim3 threads(THREADS);
	dim3 blocks(BLOCKS);
	
	// Launch the kernel
	timer_start();
	pebench<<<blocks, threads>>>(devA, devB);
	timer_stop();
	
	// Copy from the GPU
	cudaMemcpy(B, devB, size, cudaMemcpyDeviceToHost);
	
	// Clean-up and exit
	cudaFree(A);
	cudaFree(B);
	free(A);
	free(B);
	return 0;
}

// Start the timer
void timer_start() {
	cudaDeviceSynchronize();
	cudaEventCreate(&start);
	cudaEventRecord(start);
}

// End the timer
void timer_stop() {
	cudaDeviceSynchronize();
	cudaEvent_t stop;
	cudaEventCreate(&stop);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float timer = 0;
	cudaEventElapsedTime(&timer,start,stop);
	printf("Execution time: %.3lf ms \n", timer);
	float megabytes = (N*2*sizeof(unsigned)) / (1024*1024.0);
	printf("Bandwidth: %.3lf GB/s \n", megabytes/timer);
}
