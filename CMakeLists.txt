cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0104 NEW)

project(algo LANGUAGES CXX CUDA)

set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(pybind11)

# add_library(algocu STATIC
#     gpu_library.cu
# )

# set_target_properties(algocu PROPERTIES 
#     POSITION_INDEPENDENT_CODE ON
#     CUDA_VISIBILITY_PRESET "hidden"
#     # CUDA_SEPARABLE_COMPILATION ON
# )


add_library(algo MODULE
    gpu_library.cu
)

set_target_properties(algo PROPERTIES CUDA_ARCHITECTURES "61")

set_target_properties(algo PROPERTIES 
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# target_link_libraries(algo PRIVATE algocu)
target_link_libraries(algo PRIVATE pybind11::module)

pybind11_extension(algo)

target_link_libraries(algo PRIVATE pybind11::windows_extras)

# cmake_minimum_required(VERSION 3.6)
# project(CppProject LANGUAGES CXX CUDA)

# add_subdirectory(pybind11)
# pybind11_add_module(gpu_library src1.cpp)

# add_library(dummycu STATIC
#     gpu_library.cu
# )

# set_target_properties(dummycu PROPERTIES 
#     POSITION_INDEPENDENT_CODE ON
#     CUDA_VISIBILITY_PRESET "hidden"
#     # CUDA_SEPARABLE_COMPILATION ON
# )
# target_link_libraries(gpu_library PRIVATE dummycu)

# # Set CUDA and C++ flags
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
# set(CMAKE_CXX_STANDARD 11)

# # Link CUDA libraries
# target_link_libraries(gpu_library CUDA::cudart)




# find_package(Python COMPONENTS Interpreter Development)
# find_package(pybind11 CONFIG)

# # pybind11 method:
# pybind11_add_module(gpu_library src1.cpp)



# cmake_minimum_required(VERSION 3.6)
# project(CppProject)
# find_package(CUDA REQUIRED)
# find_package(Python3 3.10 REQUIRED)
# find_package(PythonLibs 3.10 REQUIRED)



# include_directories(
#     ${PYTHON_INCLUDE_DIRS}
#     ./pybind11/include
#     )

# link_directories(
#     "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64"
#     )

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# cuda_add_library(gpu_library SHARED
#   gpu_library.cu)

# target_link_libraries(gpu_library
#   ${PYTHON_LIBRARIES}
#   cudart)

# set_target_properties(gpu_library PROPERTIES PREFIX "")