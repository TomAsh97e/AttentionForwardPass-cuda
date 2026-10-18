#include <cuda_runtime.h>


#include<cmath>
#include<cstdlib>
#include <iostream>
#include <vector>
#include <random>


void checkCuda(cudaError_t error){
    if(error != cudaSuccess){
        std::cerr << "Blad CUDA: " << cudaGetErrorString(error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}
void fill(std::vector<float>& vec ,int seed){
    std::mt19937 gen(seed);
    float low = 0.2f;
    float high = 100.0f;
    std::uniform_real_distribution<float> dist(low, high);
    for(std::size_t i = 0; i < vec.size(); i++){
        vec[i] = dist(gen);
    }
}

__global__ void vectorAdd(const float* a, const float* b, float* result, int n){
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    if(i < n){
        result[i] = a[i] + b[i];
    }
}

__global__ void reduceSum(const float* a, float* res, int n){
    __shared__ float tab[256];
    const int i = threadIdx.x;
    if(i < n){
        tab[i] = a[i];
    }
    else tab[i] = 0.0f;
    __syncthreads();
    int stride = blockDim.x / 2;
    while(stride > 0){
        if(i < stride){
            tab[i] = tab[i] + tab[stride + i];  
        }
        stride /= 2;
        __syncthreads();
    }
    if(i == 0) res[0] = tab[0];
}

__global__ void reduceMax(const float* a, float* res, int n){
    __shared__ float tab[256];
    const int i = threadIdx.x;
    if(i < n){
        tab[i] = a[i];
    }
    else tab[i] = -INFINITY;
    __syncthreads();
    int stride = blockDim.x /2;

    while(stride > 0){
        if(i < stride){
            tab[i] = max(tab[i],tab[i+stride]);
        }
        stride/=2;
        __syncthreads();
    }
    if(i == 0) res[0] = tab[0];
}


int main(){
    const int n = 17;
    const std::size_t bytes = n * sizeof(float);

    // std::vector<float> a(n);
    // std::vector<float> b(n);
    // std::vector<float> ansCPU(n);
    // std::vector<float> ansGPU(n);
    // fill(a , 12345);
    // fill(b, 23456);
    std::vector<float>test(n, 1.0f);
    float* d_test = nullptr;
    checkCuda(cudaMalloc(&d_test, bytes));
    float* d_res = nullptr;
    checkCuda(cudaMalloc(&d_res, sizeof(float)));
    checkCuda(cudaMemcpy(d_test, test.data(), bytes, cudaMemcpyHostToDevice));
    //reduceSum<<<1, 256>>>(d_test, d_res ,n);
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    for(int i = 0; i < 10; ++i){
        reduceMax<<<1, 256>>>(d_test, d_res, n);
        checkCuda(cudaGetLastError());
    }
    checkCuda(cudaDeviceSynchronize());
    
    for(int i = 0; i < 20; ++i){
        checkCuda(cudaEventRecord(start));
        for(int i = 0; i < 1000; i++){
            reduceMax<<<1, 256>>>(d_test, d_res, n);
        }
    
        checkCuda(cudaEventRecord(stop));
        checkCuda(cudaGetLastError());

        checkCuda(cudaEventSynchronize(stop));

        float time;
        checkCuda(cudaEventElapsedTime(&time, start, stop));
    
        std::cout << "Kernel avg exec time: " << time/1000.0f << " ms" << std::endl;
    }
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    float result = 0.0f;
    checkCuda(cudaMemcpy(&result,d_res, sizeof(float),cudaMemcpyDeviceToHost));





    // for(std::size_t i = 0; i < ansCPU.size(); i++){
    //     ansCPU[i] = a[i] + b[i];
    // }
    // float* d_a = nullptr;
    // float* d_b = nullptr;
    // float* d_result = nullptr;

    // checkCuda(cudaMalloc(&d_a, bytes));
    // checkCuda(cudaMalloc(&d_b, bytes));
    // checkCuda(cudaMalloc(&d_result, bytes));

    // checkCuda(cudaMemcpy(d_a, a.data() , bytes, cudaMemcpyHostToDevice));
    // checkCuda(cudaMemcpy(d_b, b.data() , bytes, cudaMemcpyHostToDevice));

    // const int threadsPerBlock = 256;
    // const int blocks = (n + threadsPerBlock -1) / threadsPerBlock;

    // vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_result,n);

    
    checkCuda(cudaFree(d_test));
    checkCuda(cudaFree(d_res));
    
    if (result != 1) {
        std::cout << "Nie działa: " << 1 - result; 
        return 1;
    }
    else {
        std::cout << "git"; 
        return 0;
    }

    // checkCuda(cudaMemcpy(ansGPU.data(), d_result, bytes, cudaMemcpyDeviceToHost));

    // checkCuda(cudaFree(d_a));
    // checkCuda(cudaFree(d_b));
    // checkCuda(cudaFree(d_result));

    // for (int i = 0; i < n; ++i) {
    //     if (!std::isfinite(ansGPU[i]) ||
    //         std::abs(ansCPU[i] - ansGPU[i]) > 1e-5f) {
    //         std::cerr << "Roznica przy indeksie " << i
    //                   << ": CPU=" << ansCPU[i]
    //                   << ", GPU=" << ansGPU[i] << '\n';
    //         return EXIT_FAILURE;
    //     }
    // }

    // std::cout << "Wyniki CPU i GPU zgodne\n";


}
