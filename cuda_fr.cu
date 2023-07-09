
//system
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>
#include <assert.h>
//cuda
#include <cuda.h>
#include <cuda_runtime.h>
//thrust
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/complex.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Define a unary absolute value functor
struct abs_double {
    __host__ __device__ double operator()(double x) const { return fabs(x); }
};

// CUDA kernel for calculating repulsive forces
__global__ void calculateRepulsiveForces(const double* positions,
    double* repulsiveForces, const double k,
    const int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNodes) {
       
        double repulsiveForceX = 0.0;
        double repulsiveForceY = 0.0;

        for (int j = 0; j < numNodes; ++j) {
            if (i == j)
                continue;

            double deltaX = positions[i * 2] - positions[j * 2];
            double deltaY = positions[i * 2 + 1] - positions[j * 2 + 1];
            double distance = fmax(0.01, sqrt(deltaX * deltaX + deltaY * deltaY));
            double repulsiveForce = k * k / distance;

            repulsiveForceX += repulsiveForce * (deltaX / distance);
            repulsiveForceY += repulsiveForce * (deltaY / distance);

            // Print intermediate values
            /*if (i == 13)
                printf("i = %d, j = %d, deltaX = %.16f, deltaY = %.16f, distance = %.16f, rf = %.16f, rx=%.16f, ry = %.16f\n", i, j, deltaX, deltaY, distance, repulsiveForce, repulsiveForceX, repulsiveForceY);
        */
        }

        repulsiveForces[i * 2] = repulsiveForceX;
        repulsiveForces[i * 2 + 1] = repulsiveForceY;

        // Print final repulsive forces
        //if (i == 2)
            //printf("i = %d, repulsiveForceX = %.16f, repulsiveForceY = %.16f\n", i, repulsiveForces[i * 2], repulsiveForceY);
    }
}

__global__ void calculateAttractiveForcesSingleThread(int* edges, int numEdges, const double* positions,
    double* attractiveForces, const double k,
    const int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i==1)
        for (int i = 0; i < numEdges; ++i) {
            int node1 = edges[i * 2];
            int node2 = edges[i * 2 + 1];

            double attractiveForceX = 0.0;
            double attractiveForceY = 0.0;

            double deltaX = positions[node1 * 2] - positions[node2 * 2];
            double deltaY = positions[node1 * 2 + 1] - positions[node2 * 2 + 1];
            double distance = max(0.01, sqrt(deltaX * deltaX + deltaY * deltaY));
            double attractiveForce = distance * distance / k;

            attractiveForceX = attractiveForce * (deltaX / distance);
            attractiveForceY = attractiveForce * (deltaY / distance);

            attractiveForces[node1 * 2] -= attractiveForceX;
            attractiveForces[node1 * 2 + 1] -= attractiveForceY;
            attractiveForces[node2 * 2] += attractiveForceX;
            attractiveForces[node2 * 2 + 1] += attractiveForceY;
        }
}

// CUDA kernel for calculating attractive forces
__global__ void calculateAttractiveForces(int* edges, int numEdges, const double* positions,
    double* attractiveForces, const double k,
    const int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numEdges) {
        int node1 = edges[i * 2];
        int node2 = edges[i * 2 + 1];

        double attractiveForceX = 0.0;
        double attractiveForceY = 0.0;

        double deltaX = positions[node1 * 2] - positions[node2 * 2];
        double deltaY = positions[node1 * 2 + 1] - positions[node2 * 2 + 1];
        double distance = max(0.01, sqrt(deltaX * deltaX + deltaY * deltaY));
        double attractiveForce = distance * distance / k;

        attractiveForceX = attractiveForce * (deltaX / distance);
        attractiveForceY = attractiveForce * (deltaY / distance);

        atomicAdd(&(attractiveForces[node1 * 2]), -attractiveForceX);
        atomicAdd(&(attractiveForces[node1 * 2 + 1]), -attractiveForceY);
        atomicAdd(&(attractiveForces[node2 * 2]), attractiveForceX);
        atomicAdd(&(attractiveForces[node2 * 2 + 1]), attractiveForceY);

    }

}

// CUDA kernel for applying forces to position array
__global__ void applyForces(double* positions,
    double* attractiveForces, double* repulsiveForces, int numNodes, double temp) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 1) {

        for (int i = 0; i < numNodes; ++i) {
            double netForceX = attractiveForces[i * 2] + repulsiveForces[i * 2];
            double netForceY = attractiveForces[i * 2 + 1] + repulsiveForces[i * 2 + 1];

            double distance = max(0.01, sqrt(netForceX * netForceX + netForceY * netForceY));
            double displacementX = min(distance, temp) * netForceX / distance;
            double displacementY = min(distance, temp) * netForceY / distance;

            positions[i * 2] += displacementX;
            positions[i * 2 + 1] += displacementY;

            // Ensure node stays within bounding box
            positions[i * 2] = max(0.01, min(positions[i * 2], 1.0));
            positions[i * 2 + 1] = max(0.01, min(positions[i * 2 + 1], 1.0));
        }

        //double netForceX = attractiveForces[i * 2] + repulsiveForces[i * 2];
        //double netForceY = attractiveForces[i * 2 + 1] + repulsiveForces[i * 2 + 1];

        //double distance = fmax(0.01, sqrt(netForceX * netForceX + netForceY * netForceY));
        //double displacementX = fmin(distance, temp) * netForceX / distance;
        //double displacementY = fmin(distance, temp) * netForceY / distance;

        //positions[i * 2] += displacementX;
        //positions[i * 2 + 1] += displacementY;

        //// Ensure node stays within bounding box
        //positions[i * 2] = fmax(0.01, fmin(positions[i * 2], 1.0));
        //positions[i * 2 + 1] = fmax(0.01, fmin(positions[i * 2 + 1], 1.0));

    }
}

//Takes in edges data in the form:
    // [v1, v2, v1, v3...v45, v48]
    // where v1 is connected to v2, v1 is conneted to v3, and v45 is connected to v48
    // Returns a double array of calculated positions of the vetices
    // [x1, y1, x2, y2, .... , xn, yn]
    // where x1 is the x position for vertex 1
double* fruchterman_reingold_layout_cuda(
    int* edges, int numEdges, int numNodes, int iterations = 50,
    double k = 0.0, double temp = 1.0, double cooling_factor = 0.95, int seed = 42) {

    std::random_device rd;
    std::mt19937 gen(seed != 0 ? seed : rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    if (k == 0.0) {
        // Compute default spring constant
        double A = 1.0;
        k = sqrt(A / numNodes);
    }

    // Allocate host memory and Initialize positions randomly
    double* pos = new double[numNodes * 2];
    for (int i = 0;i < numNodes;i++) {
        pos[i * 2] = dis(gen);
        pos[i * 2 + 1] = dis(gen);

    }

    // Allocate device memory
    int* d_edges;
    double* d_positions;
    double* d_repulsiveForces;
    double* d_attractiveForces;
    gpuErrchk(cudaMalloc((void**)&d_edges, numEdges * 2 * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_edges, edges, numEdges * 2 * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**)&d_positions, numNodes * 2 * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_positions, pos, numNodes * 2 * sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**)&d_repulsiveForces, numNodes * 2 * sizeof(double)));
    gpuErrchk(cudaMemset(d_repulsiveForces, 0, numNodes * 2 * sizeof(double)));

    gpuErrchk(cudaMalloc((void**)&d_attractiveForces, numNodes * 2 * sizeof(double)));
    gpuErrchk(cudaMemset(d_attractiveForces, 0, numNodes * 2 * sizeof(double)));

    // CUDA grid and block dimensions
    int blockSize = 256;
    int gridSize = (numNodes + blockSize - 1) / blockSize;

    for (int iter = 0; iter < iterations; ++iter) {
        // Compute repulsive forces
        calculateRepulsiveForces<<<gridSize,blockSize>>>(d_positions, d_repulsiveForces, k, numNodes);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
        // Compute attractive forces
        calculateAttractiveForces<<<gridSize,blockSize>>>(d_edges, numEdges, d_positions, d_attractiveForces, k, numNodes);
        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());

        applyForces<<<gridSize,blockSize>>>(d_positions, d_attractiveForces, d_repulsiveForces, numNodes, temp);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        temp *= cooling_factor;

        // Accumulate sum of absolute forces
        double sumRepulsiveForces = 0.0;
        double sumAttractiveForces = 0.0;

        auto sumRepulsive = thrust::device_pointer_cast(d_repulsiveForces);
        auto sumAttractive = thrust::device_pointer_cast(d_attractiveForces);

        //std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1); // Set precision to maximum
        //cout << "repulsive: " << endl;
        //for (int i = 0; i < numNodes*2; i++) {
        //    std::cout << sumRepulsive[i] << " ";
        //    if (i != 0 && i % 9 == 0) cout << endl;
        //}
        //cout << "attracgtive: " << endl;
        //for (int i = 0; i < numNodes * 2; i++) {
        //    std::cout << sumAttractive[i] << " ";
        //    if (i != 0 && i % 9 == 0) cout << endl;
        //}

        ////std::cout << "repo2x: " << sumRepulsive[4] << std::endl;

        //std::cout << "repulsive forces: " << sumRepulsiveForces << std::endl;
        //std::cout << "Sum of absolute values of attractive forces: " << sumAttractiveForces << std::endl;

        //if (iter == 0) {
        //    for (int i = 0; i < 10; i++) {
        //        std::cout << sumRepulsive[i] << " ";
        //    }
        //    std::cout << std::endl;
        //}

        //reset attractive and repulsive forces
        gpuErrchk(cudaMemset(d_attractiveForces, 0, numNodes * 2 * sizeof(double)));
        gpuErrchk(cudaMemset(d_repulsiveForces, 0, numNodes * 2 * sizeof(double)));

    }

    cudaMemcpy(pos, d_positions, numNodes * 2 * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_positions);
    cudaFree(d_repulsiveForces);
    cudaFree(d_attractiveForces);

    return pos;
}
