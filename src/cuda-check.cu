#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_cuda()
{
	printf("hello cuda%d\n", threadIdx.x);
}

int main()
{
	hello_cuda<<<1,4>>>();

	cudaDeviceSynchronize();

	return 0;

}
