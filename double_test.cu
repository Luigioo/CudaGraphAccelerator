#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

//system
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>
#include <string>

using namespace std;


__global__ void c_add(double* devicePtr, double deltaX, double deltaY, double k) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double distance = fmax(0.01, sqrt(deltaX * deltaX + deltaY * deltaY));
    double repulsiveForce = k * k / distance;

    double repulsiveForceX = repulsiveForce * (deltaX / distance);
    double repulsiveForceY = repulsiveForce * (deltaY / distance);
    // Print intermediate values
    // log("k = %.16f, deltaX = %.16f, deltaY = %.16f, distance = %.16f, rf = %.16f, rx=%.16f, ry = %.16f\n", k, deltaX, deltaY, distance, repulsiveForce, repulsiveForceX, repulsiveForceY);
    // log("asdf");
    *devicePtr = (deltaX * deltaX + deltaY * deltaY);
}


void add(double deltaX, double deltaY, double k) {

    double distance = std::max(0.01, std::sqrt(deltaX * deltaX + deltaY * deltaY));
    double repulsiveForce = k * k / distance;

    double repulsiveForceX = repulsiveForce * (deltaX / distance);
    double repulsiveForceY = repulsiveForce * (deltaY / distance);

    // Print intermediate values
    printf("tosqrt=%.16f, deltaX = %.16f, deltaY = %.16f, distance = %.16f, rf = %.16f, rx=%.16f, ry = %.16f\n", (deltaX * deltaX + deltaY * deltaY), deltaX, deltaY, distance, repulsiveForce, repulsiveForceX, repulsiveForceY);
}

void run_test() {
    printf("ddd");

    double deltaX = -0.5279227649178649;
    double deltaY = -0.1327964198334066;

    double A = 1.0;
    double k = sqrt(A / 50);

    double myVariable;

    double* devicePtr;
    cudaMalloc(&devicePtr, sizeof(double));

    c_add <<<1, 1>>> (devicePtr, deltaX, deltaY, k);

    cudaMemcpy(&myVariable, devicePtr, sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(devicePtr, &myVariable, sizeof(double), cudaMemcpyHostToDevice);

    printf("%.16f", myVariable);
    printf("\n");

    add(deltaX, deltaY, k);

}
