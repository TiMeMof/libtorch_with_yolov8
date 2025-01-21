#include <stdio.h>

// 打印CUDA版本信息的函数
void printCudaVersion() {
    cudaError_t err = cudaSuccess;
    int runtimeVersion = 0;
    err = cudaRuntimeGetVersion(&runtimeVersion);
    if (err == cudaSuccess) {
        printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
    } else {
        printf("Error getting runtime version: %s\n", cudaGetErrorString(err));
    }

    int driverVersion = 0;
    err = cudaDriverGetVersion(&driverVersion);
    if (err == cudaSuccess) {
        printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 1000) / 10);
    } else {
        printf("Error getting driver version: %s\n", cudaGetErrorString(err));
    }
}


__global__ void myKernel() 
{
    printf("Hello, world from the device!\n"); 
} 

int main() 
{ 
    printCudaVersion();
    myKernel<<<4,4>>>(); 
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        return 1;
        }
        else {
            printf("No CUDA error\n");
            }
     cudaDeviceSynchronize();
} 

