#ifndef GPULIBRARY_H
#define GPULIBRARY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

void processWrapper(pybind11::array_t<int> array, int numNodes);

#endif