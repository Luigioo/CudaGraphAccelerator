//system
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>
//cuda
#include <cuda_runtime.h>

namespace normal_fr{


    
void calculateRepulsiveForces(const double* positions, double* repulsiveForces, const double k, const int numNodes) {
    for (int i = 0; i < numNodes; ++i) {
        double repulsiveForceX = 0.0;
        double repulsiveForceY = 0.0;

        for (int j = 0; j < numNodes; ++j) {
            if (i == j)
                continue;

            double deltaX = positions[i * 2] - positions[j * 2];
            double deltaY = positions[i * 2 + 1] - positions[j * 2 + 1];
            double distance = std::max(0.01, std::sqrt(deltaX * deltaX + deltaY * deltaY));
            double repulsiveForce = k * k / distance;

            repulsiveForceX += repulsiveForce * (deltaX / distance);
            repulsiveForceY += repulsiveForce * (deltaY / distance);

            // Print intermediate values
            if(i==2)
                printf("i = %d, j = %d, deltaX = %.16f, deltaY = %.16f, distance = %.16f, rf = %.16f, rx=%.16f, ry = %.16f\n", i, j, deltaX, deltaY, distance, repulsiveForce, repulsiveForceX, repulsiveForceY);
        }

        repulsiveForces[i * 2] = repulsiveForceX;
        repulsiveForces[i * 2 + 1] = repulsiveForceY;

        // Print final repulsive forces
        if(i==2)
            printf("i = %d, repulsiveForceX = %.16lf, repulsiveForceY = %.16lf\n", i, repulsiveForceX, repulsiveForceY);
    }
}

    void calculateAttractiveForces(int* edges, int numEdges, const double* positions, double* attractiveForces,
                                const double k, const int numNodes) {
        for (int i = 0; i < numEdges; ++i) {
            int node1 = edges[i * 2];
            int node2 = edges[i * 2 + 1];

            double attractiveForceX = 0.0;
            double attractiveForceY = 0.0;

            double deltaX = positions[node1 * 2] - positions[node2 * 2];
            double deltaY = positions[node1 * 2 + 1] - positions[node2 * 2 + 1];
            double distance = std::max(0.01, std::sqrt(deltaX * deltaX + deltaY * deltaY));
            double attractiveForce = distance * distance / k;

            attractiveForceX = attractiveForce * (deltaX / distance);
            attractiveForceY = attractiveForce * (deltaY / distance);

            attractiveForces[node1 * 2] -= attractiveForceX;
            attractiveForces[node1 * 2 + 1] -= attractiveForceY;
            attractiveForces[node2 * 2] += attractiveForceX;
            attractiveForces[node2 * 2 + 1] += attractiveForceY;
        }
    }

    void applyForces(double* positions, double* attractiveForces, double* repulsiveForces,
                    int numNodes, double temp) {
        for (int i = 0; i < numNodes; ++i) {
            double netForceX = attractiveForces[i * 2] + repulsiveForces[i * 2];
            double netForceY = attractiveForces[i * 2 + 1] + repulsiveForces[i * 2 + 1];

            double distance = std::max(0.01, std::sqrt(netForceX * netForceX + netForceY * netForceY));
            double displacementX = std::min(distance, temp) * netForceX / distance;
            double displacementY = std::min(distance, temp) * netForceY / distance;

            positions[i * 2] += displacementX;
            positions[i * 2 + 1] += displacementY;

            // Ensure node stays within bounding box
            positions[i * 2] = std::max(0.01, std::min(positions[i * 2], 1.0));
            positions[i * 2 + 1] = std::max(0.01, std::min(positions[i * 2 + 1], 1.0));
        }
    }

    double* fruchterman_reingold_layout_cuda(int* edges, int numEdges, int numNodes,
                                            int iterations = 50, double k = 0.0, double temp = 1.0,
                                            double cooling_factor = 0.95, int seed = 42) {
        std::random_device rd;
        std::mt19937 gen(seed != 0 ? seed : rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        if (k == 0.0) {
            // Compute default spring constant
            double A = 1.0;
            k = std::sqrt(A / numNodes);
        }

        // Allocate host memory and initialize positions randomly
        double* pos = new double[numNodes * 2];
        for (int i = 0; i < numNodes; i++) {
            pos[i * 2] = dis(gen);
            pos[i * 2 + 1] = dis(gen);
        }

        double* repulsiveForces = new double[numNodes * 2]();
        double* attractiveForces = new double[numNodes * 2]();

        for (int iter = 0; iter < iterations; ++iter) {
            // Compute repulsive forces
            calculateRepulsiveForces(pos, repulsiveForces, k, numNodes);

            // Compute attractive forces
            calculateAttractiveForces(edges, numEdges, pos, attractiveForces, k, numNodes);

            applyForces(pos, attractiveForces, repulsiveForces, numNodes, temp);


            //Accumulate sum of absolute forces
            double sumRepulsiveForces = 0.0;
            double sumAttractiveForces = 0.0;
            for (int i = 0; i < numNodes * 2; ++i) {
                sumRepulsiveForces += std::abs(repulsiveForces[i]);
                sumAttractiveForces += std::abs(attractiveForces[i]);
            }
            std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1); // Set precision to maximum
            printf("repo2x: %.16lf", repulsiveForces[4]);
            std::cout << "Sum of Repulsive Forces: " << sumRepulsiveForces << std::endl;
            std::cout << "Sum of Attractive Forces   : " << sumAttractiveForces << std::endl;
            
            if(iter==0){
             for (int i = 0; i < 10; i++) {
                std::cout << repulsiveForces[i] << " ";
             }    
            }

            // Reset attractive and repulsive forces
            std::fill(repulsiveForces, repulsiveForces + (numNodes * 2), 0.0);
            std::fill(attractiveForces, attractiveForces + (numNodes * 2), 0.0);

            temp *= cooling_factor;


        }

        delete[] repulsiveForces;
        delete[] attractiveForces;

        return pos;
    }

}
