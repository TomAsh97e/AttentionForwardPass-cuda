#include <iostream>

void checkCuda(cudaError_t error){
    if(error != cudaSuccess){
        std::cerr << "Blad CUDA: " << cudaGetErrorString(error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}
__global__ void kernel_qk(const float* q, const float* k, float* s, int n, int d){
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    if(l < n && m < n){
        float tmp = 0;
        for(int i = 0 ; i < d; ++i){
            tmp += q[l*d+i] * k[m*d+i];
        }
        tmp /= sqrtf(static_cast<float>(d));
        s[l*n+m] = tmp;
    }
    
}

__global__ void kernel_softmax(const float* s, float* p, int n){
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if( l < n){
        float m{-INFINITY};
        for(int i = 0; i < n; ++i){
            m = max(m , s[l*n + i]);
        }
        float z {};
        for(int i = 0; i < n; ++i){
            z += expf(s[l*n + i] - m);
        }
        for(int i = 0; i <  n; ++i){
            p[l*n +i] = expf(s[l*n+i] - m)/ z;
        }
    }
}

__global__ void kernel_pv(const float* p, const float* v, float* o, int n, int d){
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    if(m < d  && l < n){
        float tmp{};
        for(int i = 0; i < n; ++i){
            tmp += p[l*n + i] * v[i*d+ m];
        }
        o[l*d + m] = tmp;
    }
}

void sft (const float* q, const float* k,const float* v, float* s,float* p, float* o,  int n, int d){
    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_s = nullptr;
    float* d_p = nullptr;
    float* d_v = nullptr;
    float* d_o = nullptr;

    checkCuda(cudaMalloc(&d_q, n*d*sizeof(float)));
    checkCuda(cudaMalloc(&d_k, n*d*sizeof(float)));
    checkCuda(cudaMalloc(&d_s, n*n*sizeof(float)));
    checkCuda(cudaMalloc(&d_p, n*n*sizeof(float)));
    checkCuda(cudaMalloc(&d_o, n*d*sizeof(float)));
    checkCuda(cudaMalloc(&d_v, n*d*sizeof(float)));

    checkCuda(cudaMemcpy(d_q, q, n*d*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_k, k, n*d*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_v, v, n*d*sizeof(float),cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((n+ block.x -1) / block.x, (n + block.y-1) / block.y);


    kernel_qk<<<grid, block>>>(d_q, d_k, d_s, n, d);
    
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(s, d_s, n*n*sizeof(float), cudaMemcpyDeviceToHost));

    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock -1) / threadsPerBlock;
    kernel_softmax<<<blocks, threadsPerBlock>>>(d_s, d_p, n);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(p, d_p, n*n*sizeof(float), cudaMemcpyDeviceToHost));

    dim3 grid1((d+block.x-1) / block.x,(n + block.y-1)/block.y);
    kernel_pv<<<grid1, block>>>(d_p, d_v, d_o, n ,d);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(o, d_o, n*d*sizeof(float), cudaMemcpyDeviceToHost));
    

    checkCuda(cudaFree(d_q));
    checkCuda(cudaFree(d_k));
    checkCuda(cudaFree(d_s));
    checkCuda(cudaFree(d_p));
    checkCuda(cudaFree(d_o));
    checkCuda(cudaFree(d_v));
}