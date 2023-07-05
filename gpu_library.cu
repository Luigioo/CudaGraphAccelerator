
//pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
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

//local
#include "cuda_fr.cu"
#include "normal_fr.cu"
#include "double_test.cu"

namespace py = pybind11;


py::array_t<double> processWrapper(py::array_t<int> array, int numNodes, int iteration) {
    py::buffer_info info = array.request(); // get a pointer to the array buffer
    int* ptr = static_cast<int*>(info.ptr);
    int numEdges = info.size/2; 
    double* positions = normal_fr::fruchterman_reingold_layout_cuda(ptr, numEdges, numNodes, iteration);

    py::array_t<double> result({numNodes*2}, {sizeof(double)});
    
    // Get a pointer to the underlying data buffer of the NumPy array
    double* result_ptr = static_cast<double*>(result.request().ptr);

    // Copy the elements from the existing array to the NumPy array
    std::copy(positions, positions + numNodes*2, result_ptr);
    return result;
}

py::array_t<double> processWrapperCuda(py::array_t<int> array, int numNodes, int iteration) {
    py::buffer_info info = array.request(); // get a pointer to the array buffer
    int* ptr = static_cast<int*>(info.ptr);
    int numEdges = info.size/2; 
    double* positions = fruchterman_reingold_layout_cuda(ptr, numEdges, numNodes, iteration);

    py::array_t<double> result({numNodes*2}, {sizeof(double)});
    
    // Get a pointer to the underlying data buffer of the NumPy array
    double* result_ptr = static_cast<double*>(result.request().ptr);

    // Copy the elements from the existing array to the NumPy array
    std::copy(positions, positions + numNodes*2, result_ptr);
    return result;
}




PYBIND11_MODULE(algo, m)
{
  m.def("fr", &processWrapper);
  m.def("fr_cuda", &processWrapperCuda);
  m.def("double_test", &run_test);
}