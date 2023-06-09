#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

using Graph = std::unordered_map<int, std::vector<int>>;

std::unordered_map<int, std::vector<double>> fruchterman_reingold_layout(Graph& G, int iterations = 50, double k = 0.0, double temp = 1.0, double cooling_factor = 0.95, int seed = 0) {
    std::random_device rd;
    std::mt19937 gen(seed != 0 ? seed : rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    if (k == 0.0) {
        // Compute default spring constant
        int n = G.size();
        double A = 1.0;
        k = std::sqrt(A / n);
    }

    // Initialize positions randomly
    std::unordered_map<int, std::vector<double>> pos;
    for (const auto& pair : G) {
        int node = pair.first;
        pos[node] = { dis(gen), dis(gen) };
    }

    for (int i = 0; i < iterations; ++i) {
        // Compute repulsive forces between all pairs of nodes
        std::unordered_map<int, std::vector<double>> repulsive_forces;
        for (const auto& pair1 : G) {
            int node1 = pair1.first;
            repulsive_forces[node1] = { 0.0, 0.0 };
            for (const auto& pair2 : G) {
                int node2 = pair2.first;
                if (node1 == node2)
                    continue;
                std::vector<double> delta = { pos[node1][0] - pos[node2][0], pos[node1][1] - pos[node2][1] };
                double distance = std::max(0.01, std::sqrt(delta[0] * delta[0] + delta[1] * delta[1]));
                double repulsive_force = k * k / distance;
                repulsive_forces[node1][0] += repulsive_force * (delta[0] / distance);
                repulsive_forces[node1][1] += repulsive_force * (delta[1] / distance);
            }
        }
        // Compute attractive forces between adjacent nodes
        std::unordered_map<int, std::vector<double>> attractive_forces;
        for (const auto& pair : G) {
            int node1 = pair.first;
            attractive_forces[node1] = { 0.0, 0.0 };
            for (int node2 : pair.second) {
                if (attractive_forces.find(node2) == attractive_forces.end()) {
                    // Key doesn't exist, initialize the value
                    attractive_forces[node2] = { 0.0, 0.0 };
                }
                std::vector<double> delta = { pos[node1][0] - pos[node2][0], pos[node1][1] - pos[node2][1] };
                double distance = std::max(0.01, std::sqrt(delta[0] * delta[0] + delta[1] * delta[1]));
                double attractive_force = distance * distance / k;
                attractive_forces[node1][0] -= attractive_force * delta[0] / distance;
                attractive_forces[node1][1] -= attractive_force * delta[1] / distance;
                attractive_forces[node2][0] += attractive_force * delta[0] / distance;
                attractive_forces[node2][1] += attractive_force * delta[1] / distance;
            }
        }

        // Compute new positions based on forces
        for (const auto& pair : G) {
            int node = pair.first;
            // Compute net force on node
            std::vector<double> net_force = { attractive_forces[node][0] + repulsive_forces[node][0],
                               attractive_forces[node][1] + repulsive_forces[node][1] };
            // Compute new position of node
            double distance = std::max(0.01, std::sqrt(net_force[0] * net_force[0] + net_force[1] * net_force[1]));
            std::vector<double> displacement = { std::min(distance, temp) * net_force[0] / distance, std::min(distance, temp) * net_force[1] / distance };
            pos[node][0] += displacement[0];
            pos[node][1] += displacement[1];
            // Ensure node stays within bounding box
            pos[node][0] = std::max(0.01, std::min(pos[node][0], 1.0));
            pos[node][1] = std::max(0.01, std::min(pos[node][1], 1.0));
        }

        // Reduce temperature
        temp *= cooling_factor;

        std::cout << "Iteration " << i << std::endl;
    }

    return pos;
}

#include <iostream>
#include <random>
#include <vector>
#include <unordered_map>

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
            double distance = max(0.01, sqrt(deltaX * deltaX + deltaY * deltaY));
            double repulsiveForce = k * k / distance;

            repulsiveForceX += repulsiveForce * (deltaX / distance);
            repulsiveForceY += repulsiveForce * (deltaY / distance);
        }

        repulsiveForces[i * 2] = repulsiveForceX;
        repulsiveForces[i * 2 + 1] = repulsiveForceY;
    }
}

// CUDA kernel for calculating attractive forces
__global__ void calculateAttractiveForces(int* edges, int numEdges, const double* positions,
                                          double* attractiveForces, const double k,
                                          const int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

	int node1 = edges[i*2];
	int node2 = edges[i*2+1];

	double attractiveForceX = 0.0;
	double attractiveForceY = 0.0;

	double deltaX = positions[node1 * 2] - positions[node2 * 2];
	double deltaY = positions[node1 * 2 + 1] - positions[node2 * 2 + 1];
	double distance = max(0.01, sqrt(deltaX * deltaX + deltaY * deltaY));
	double attractiveForce = distance * distance / k;

	attractiveForceX -= attractiveForce * (deltaX / distance);
	attractiveForceY -= attractiveForce * (deltaY / distance);

	attractiveForces[node1 * 2] += -attractiveForce * (deltaX / distance);
	attractiveForces[node1 * 2 + 1] += -attractiveForce * (deltaY / distance);
	attractiveForces[node2 * 2] += attractiveForce * (deltaX / distance);
	attractiveForces[node2 * 2 + 1] += attractiveForce * (deltaY / distance);

}

// CUDA kernel for applying forces to position array
__global__ void applyForces(double* positions,
							double* attractiveForces, double* repulsiveForces, int numNodes, double temp) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numNodes) {
        double netForceX = attractiveForces[i * 2] + repulsiveForces[i * 2];
		double netForceY = attractiveForces[i * 2 + 1] + repulsiveForces[i * 2 + 1];

		double distance = max(0.01, sqrt(netForceX * netForceX + netForceY * netForceY));
		double displacementX = min(distance, temp) * netForceX / distance;
		double displacementY = min(distance, temp) * netForceY / distance;

		positions[i*2] += displacementX;
		positions[i*2+1] += displacementY;

		// Ensure node stays within bounding box
		positions[i*2] = max(0.01, min(positions[i*2], 1.0));
		positions[i*2+1] = max(0.01, min(positions[i*2+1], 1.0));
    }
}

double* fruchterman_reingold_layout_cuda(
    int* edges, int numEdges, int numNodes, int iterations = 50,
    double k = 0.0, double temp = 1.0, double cooling_factor = 0.95, int seed = 0) {

    std::random_device rd;
    std::mt19937 gen(seed != 0 ? seed : rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    if (k == 0.0) {
        // Compute default spring constant
        double A = 1.0;
        k = sqrt(A / numNodes);
    }

    // Initialize positions randomly
    double *pos = new double[numNodes*2];
    for (int i=0;i<numNodes;i++) {
        pos[i*2] = dis(gen);
        pos[i*2+1] = dis(gen);
		
    }

    // Allocate device memory

	int* d_edges;
    cudaMalloc((void**)&d_edges, numEdges * 2 * sizeof(int));
    cudaMemcpy(d_edges, edges, numEdges * sizeof(int), cudaMemcpyHostToDevice);

    double* d_positions;
    cudaMalloc((void**)&d_positions, numNodes * 2 * sizeof(double));
    cudaMemcpy(d_positions, pos, numNodes * sizeof(double), cudaMemcpyHostToDevice);

    double* d_repulsiveForces;
    cudaMalloc((void**)&d_repulsiveForces, numNodes * 2 * sizeof(double));

    double* d_attractiveForces;
    cudaMalloc((void**)&d_attractiveForces, numNodes * 2 * sizeof(double));

    // Copy data from host to device
	// // Get the size of the unordered map
    // size_t mapSize = G.size();

    // // Convert sizes to device array
    // int* deviceSizes;
    // cudaMalloc(&deviceSizes, mapSize * sizeof(int));
    // cudaMemcpy(deviceSizes, G.keys(), mapSize * sizeof(int), cudaMemcpyHostToDevice);

    // // Convert vectors to device array of pointers
    // int** deviceVectors;
    // cudaMalloc(&deviceVectors, mapSize * sizeof(int*));
    // std::vector<int*> hostVectors;
    // for (const auto& pair : G)
    // {
    //     std::vector<int>& vec = pair.second;
    //     int* deviceVector;
    //     cudaMalloc(&deviceVector, vec.size() * sizeof(int));
    //     cudaMemcpy(deviceVector, vec.data(), vec.size() * sizeof(int), cudaMemcpyHostToDevice);
    //     hostVectors.push_back(deviceVector);
    // }
    // cudaMemcpy(deviceVectors, hostVectors.data(), mapSize * sizeof(int*), cudaMemcpyHostToDevice);

    // cudaMemcpy(d_nodes, &G[0], numNodes * sizeof(int), cudaMemcpyHostToDevice);

    // cudaMemcpy(d_positions, initialPositions, numNodes * 2 * sizeof(double), cudaMemcpyHostToDevice);

    // CUDA grid and block dimensions
    int blockSize = 256;
    int gridSize = (numNodes + blockSize - 1) / blockSize;

    for (int iter = 0; iter < iterations; ++iter) {
        // Compute repulsive forces
        calculateRepulsiveForces<<<gridSize, blockSize>>>(d_positions, d_repulsiveForces, k, numNodes);

        // Compute attractive forces
        calculateAttractiveForces<<<gridSize, blockSize>>>(d_edges, numEdges, d_positions, d_attractiveForces, k, numNodes);

		applyForces<<<gridSize, blockSize>>>(d_positions, d_attractiveForces, d_repulsiveForces, numNodes, temp);

        temp *= cooling_factor;

        // // Copy forces from device to host
        // double* repulsiveForces = new double[numNodes * 2];
        // cudaMemcpy(repulsiveForces, d_repulsiveForces, numNodes * 2 * sizeof(double), cudaMemcpyDeviceToHost);

        // double* attractiveForces = new double[numNodes * 2];
        // cudaMemcpy(attractiveForces, d_attractiveForces, numNodes * 2 * sizeof(double), cudaMemcpyDeviceToHost);

        // // Compute new positions based on forces
        // for (int i = 0; i < numNodes; ++i) {
        //     double netForceX = attractiveForces[i * 2] + repulsiveForces[i * 2];
        //     double netForceY = attractiveForces[i * 2 + 1] + repulsiveForces[i * 2 + 1];

        //     double distance = max(0.01, sqrt(netForceX * netForceX + netForceY * netForceY));
        //     double displacementX = min(distance, temp) * netForceX / distance;
        //     double displacementY = min(distance, temp) * netForceY / distance;

        //     pos[i][0] += displacementX;
        //     pos[i][1] += displacementY;

        //     // Ensure node stays within bounding box
        //     pos[i][0] = max(0.01, min(pos[i][0], 1.0));
        //     pos[i][1] = max(0.01, min(pos[i][1], 1.0));
        // }


        // delete[] repulsiveForces;
        // delete[] attractiveForces;
        // Reduce temperature
    }

    cudaMemcpy(pos, d_positions, numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_positions);
    cudaFree(d_repulsiveForces);
    cudaFree(d_attractiveForces);



    return pos;
}

void processWrapper(pybind11::array_t<int> array, int numNodes) {
    pybind11::buffer_info info = array.request();
    int* ptr = static_cast<int*>(info.ptr);
    int size = info.size;
    fruchterman_reingold_layout_cuda(ptr, size, numNodes);
}

PYBIND11_MODULE(gpu_library, m)
{
  m.def("fr", &fruchterman_reingold_layout);
  m.def("fr_cuda", &processWrapper);
}