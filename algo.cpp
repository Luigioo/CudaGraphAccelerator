#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>

namespace py = pybind11;

float foo(float a, float b){
    return a+b;
}


struct Point {
    double x;
    double y;
};

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

std::vector<int> typetest(const std::vector<int>& input) {
    std::vector<int> output;
    output.reserve(input.size());

    for (int element : input) {
        output.push_back(element + 1);
    }

    return output;
}


PYBIND11_MODULE(algo, handle){
    handle.doc() = "This is the module docs....";
    handle.def("fruchterman_reingold_layout", &fruchterman_reingold_layout);
    handle.def("foo", &foo);
    // handle.def("typetest", &)
}

// int main(int argc, char const *argv[])
// {
//     // Create an instance of the graph
//     Graph myGraph;

//     // Add vertices and their adjacent vertices
//     myGraph[0] = {1, 2, 3};   // Vertex 0 is connected to vertices 1, 2, and 3
//     myGraph[1] = {2, 4};      // Vertex 1 is connected to vertices 2 and 4
//     myGraph[2] = {3, 4};      // Vertex 2 is connected to vertices 3 and 4
//     myGraph[3] = {0};         // Vertex 3 is connected to vertex 0
//     myGraph[4] = {1};          // Vertex 4 has no adjacent vertices
//     fruchterman_reingold_layout(myGraph);
//     return 0;
// }
