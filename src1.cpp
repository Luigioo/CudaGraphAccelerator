#include "gpu_library.h"



void foo(){
	printf("lkjl");
}

PYBIND11_MODULE(gpu_library, m)
{
  m.def("fr", &fruchterman_reingold_layout);
  m.def("fr_cuda", &processWrapper);
  m.def("foo", &foo);
}
