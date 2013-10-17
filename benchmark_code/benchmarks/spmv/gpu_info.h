#include <endian.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <inttypes.h>


void compute_active_thread(unsigned int *thread,
					unsigned int *grid,
					int task,
					int pad,
					int major,
					int minor,
					int warp_size,
					int sm);