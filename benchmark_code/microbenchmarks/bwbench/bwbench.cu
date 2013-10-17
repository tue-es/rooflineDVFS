
// Includes
#include <stdio.h>

// Constants
#define WPT 64
#define THREADS 512
#define BLOCKS 14*2*8
#define N (WPT*BLOCKS*THREADS)

// Kernel
__global__ void bwbench(unsigned *A, unsigned *B) {
	unsigned i = blockIdx.x*THREADS + threadIdx.x;
	B[i+ 0*THREADS*BLOCKS] = A[i+ 0*THREADS*BLOCKS];
	B[i+ 1*THREADS*BLOCKS] = A[i+ 1*THREADS*BLOCKS];
	B[i+ 2*THREADS*BLOCKS] = A[i+ 2*THREADS*BLOCKS];
	B[i+ 3*THREADS*BLOCKS] = A[i+ 3*THREADS*BLOCKS];
	B[i+ 4*THREADS*BLOCKS] = A[i+ 4*THREADS*BLOCKS];
	B[i+ 5*THREADS*BLOCKS] = A[i+ 5*THREADS*BLOCKS];
	B[i+ 6*THREADS*BLOCKS] = A[i+ 6*THREADS*BLOCKS];
	B[i+ 7*THREADS*BLOCKS] = A[i+ 7*THREADS*BLOCKS];
	B[i+ 8*THREADS*BLOCKS] = A[i+ 8*THREADS*BLOCKS];
	B[i+ 9*THREADS*BLOCKS] = A[i+ 9*THREADS*BLOCKS];
	B[i+10*THREADS*BLOCKS] = A[i+10*THREADS*BLOCKS];
	B[i+11*THREADS*BLOCKS] = A[i+11*THREADS*BLOCKS];
	B[i+12*THREADS*BLOCKS] = A[i+12*THREADS*BLOCKS];
	B[i+13*THREADS*BLOCKS] = A[i+13*THREADS*BLOCKS];
	B[i+14*THREADS*BLOCKS] = A[i+14*THREADS*BLOCKS];
	B[i+15*THREADS*BLOCKS] = A[i+15*THREADS*BLOCKS];
	B[i+16*THREADS*BLOCKS] = A[i+16*THREADS*BLOCKS];
	B[i+17*THREADS*BLOCKS] = A[i+17*THREADS*BLOCKS];
	B[i+18*THREADS*BLOCKS] = A[i+18*THREADS*BLOCKS];
	B[i+19*THREADS*BLOCKS] = A[i+19*THREADS*BLOCKS];
	B[i+20*THREADS*BLOCKS] = A[i+20*THREADS*BLOCKS];
	B[i+21*THREADS*BLOCKS] = A[i+21*THREADS*BLOCKS];
	B[i+22*THREADS*BLOCKS] = A[i+22*THREADS*BLOCKS];
	B[i+23*THREADS*BLOCKS] = A[i+23*THREADS*BLOCKS];
	B[i+24*THREADS*BLOCKS] = A[i+24*THREADS*BLOCKS];
	B[i+25*THREADS*BLOCKS] = A[i+25*THREADS*BLOCKS];
	B[i+26*THREADS*BLOCKS] = A[i+26*THREADS*BLOCKS];
	B[i+27*THREADS*BLOCKS] = A[i+27*THREADS*BLOCKS];
	B[i+28*THREADS*BLOCKS] = A[i+28*THREADS*BLOCKS];
	B[i+29*THREADS*BLOCKS] = A[i+29*THREADS*BLOCKS];
	B[i+30*THREADS*BLOCKS] = A[i+30*THREADS*BLOCKS];
	B[i+31*THREADS*BLOCKS] = A[i+31*THREADS*BLOCKS];
	B[i+32*THREADS*BLOCKS] = A[i+32*THREADS*BLOCKS];
	B[i+33*THREADS*BLOCKS] = A[i+33*THREADS*BLOCKS];
	B[i+34*THREADS*BLOCKS] = A[i+34*THREADS*BLOCKS];
	B[i+35*THREADS*BLOCKS] = A[i+35*THREADS*BLOCKS];
	B[i+36*THREADS*BLOCKS] = A[i+36*THREADS*BLOCKS];
	B[i+37*THREADS*BLOCKS] = A[i+37*THREADS*BLOCKS];
	B[i+38*THREADS*BLOCKS] = A[i+38*THREADS*BLOCKS];
	B[i+39*THREADS*BLOCKS] = A[i+39*THREADS*BLOCKS];
	B[i+40*THREADS*BLOCKS] = A[i+40*THREADS*BLOCKS];
	B[i+41*THREADS*BLOCKS] = A[i+41*THREADS*BLOCKS];
	B[i+42*THREADS*BLOCKS] = A[i+42*THREADS*BLOCKS];
	B[i+43*THREADS*BLOCKS] = A[i+43*THREADS*BLOCKS];
	B[i+44*THREADS*BLOCKS] = A[i+44*THREADS*BLOCKS];
	B[i+45*THREADS*BLOCKS] = A[i+45*THREADS*BLOCKS];
	B[i+46*THREADS*BLOCKS] = A[i+46*THREADS*BLOCKS];
	B[i+47*THREADS*BLOCKS] = A[i+47*THREADS*BLOCKS];
	B[i+48*THREADS*BLOCKS] = A[i+48*THREADS*BLOCKS];
	B[i+49*THREADS*BLOCKS] = A[i+49*THREADS*BLOCKS];
	B[i+50*THREADS*BLOCKS] = A[i+50*THREADS*BLOCKS];
	B[i+51*THREADS*BLOCKS] = A[i+51*THREADS*BLOCKS];
	B[i+52*THREADS*BLOCKS] = A[i+52*THREADS*BLOCKS];
	B[i+53*THREADS*BLOCKS] = A[i+53*THREADS*BLOCKS];
	B[i+54*THREADS*BLOCKS] = A[i+54*THREADS*BLOCKS];
	B[i+55*THREADS*BLOCKS] = A[i+55*THREADS*BLOCKS];
	B[i+56*THREADS*BLOCKS] = A[i+56*THREADS*BLOCKS];
	B[i+57*THREADS*BLOCKS] = A[i+57*THREADS*BLOCKS];
	B[i+58*THREADS*BLOCKS] = A[i+58*THREADS*BLOCKS];
	B[i+59*THREADS*BLOCKS] = A[i+59*THREADS*BLOCKS];
	B[i+60*THREADS*BLOCKS] = A[i+60*THREADS*BLOCKS];
	B[i+61*THREADS*BLOCKS] = A[i+61*THREADS*BLOCKS];
	B[i+62*THREADS*BLOCKS] = A[i+62*THREADS*BLOCKS];
	B[i+63*THREADS*BLOCKS] = A[i+63*THREADS*BLOCKS];
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
	bwbench<<<blocks, threads>>>(devA, devB);
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
	float megabytes = (N*2*sizeof(unsigned)) / (1024*1024);
	printf("Bandwidth: %.3lf GB/s \n", megabytes/timer);
}
