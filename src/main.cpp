
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include "attention.hpp"

float dot_product(const std::vector<float>& q, const std::vector<float>& k, int d, int m, int l){
    float res = 0;
    for(int i = 0;i < d; ++i){
        res += q[l*d + i]*k[m*d+i];
    }
    return res;
}

float pv (const std::vector<float>& p, const std::vector<float>& v, int d, int n, int m, int l){
    float res = 0;
    for(int i = 0;i < n; ++i){
        res += p[m*n + i]*v[i*d+l];
    }
    return res;
}

void fill(std::vector<float>& vec ,int seed){
    std::mt19937 gen(seed);
    float low = -1.0f;
    float high = 1.0f;
    std::uniform_real_distribution<float> dist(low, high);
    for(std::size_t i = 0; i < vec.size(); i++){
        vec[i] = dist(gen);
    }
}

int main(){
    const int  N = 17;
    const int d = 64;
    std::vector<float> q(N*d);
    std::vector<float> k(N*d);
    std::vector<float> v(N*d);
    
    fill(q, 12345);
    fill(k, 23456);
    fill(v, 34567);


    std::vector<float> s_c(N*N);
    std::vector<float> s_g(N*N);
    std::vector<float> p_c(N*N);
    std::vector<float> p_g(N*N);
    std::vector<float> o_c(N*d);
    std::vector<float> o_g(N*d);


    float sqrt_d = sqrt(d);
    for(int l = 0; l < N; l++){
        for(int m = 0; m < N; m++){
            s_c[l*N + m] = dot_product(q, k, d, m, l)/sqrt_d;
        }
    }
    sft(q.data(), k.data(),v.data(), s_g.data(),p_g.data(),o_g.data(), N, d);
    // for(int i = 0; i < N;i++){
    //     for(int j = 0; j < N; j++){
    //         std::cout << "CPU: "<<s_c[i*N+j] << "|GPU: "<<s_g[i*N+j] <<"    "; 
    //         if(std::abs(s_c[i*N+j]-s_g[i*N+j]) > 0.001f){
    //              std::cerr<< "Rozne dane";
    //              return 1;
    //             }
    //     }
    //     std::cout << std::endl;
    // }

    
    for(int l = 0; l < N; ++l){
        float m = *std::max_element(s_c.begin()+N*l, s_c.begin()+N*l+N);
        float z{0};
        for(int i =0; i < N; ++i){
            z += std::exp(s_c[l*N+i] - m);
        }
        for(int i = 0; i < N; ++i){
            p_c[l*N+i] = std::exp(s_c[l*N+i]-m)/z;
        }
    }

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < d; ++j){
            o_c[i*d + j] = pv(p_c, v, d, N, i ,j);
        }
    }
    // for(int i = 0; i < N;i++){
    //     for(int j = 0; j < N; j++){
    //         std::cout << "CPU: "<<p_c[i*N+j] << "|GPU: "<<p_g[i*N+j] <<"    "; 
    //         if(std::abs(p_c[i*N+j]-p_g[i*N+j]) > 0.001f){
    //              std::cerr<< "Rozne dane";
    //              return 1;
    //             }
    //     }
    //     std::cout << std::endl;
    // }
    for(int i = 0; i < N;i++){
        for(int j = 0; j < d; j++){
            std::cout << "CPU: "<<o_c[i*d+j] << "|GPU: "<<o_g[i*d+j] <<"    "; 
        }
        std::cout << std::endl;
    }





    return 0;

}
