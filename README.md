
# Force Directed Algorithm on GPU (Fruchterman Reingold)

## Overview
This project aims to accelerate the Fruchterman Reingold algorithm using CUDA on a GPU. The Fruchterman Reingold algorithm is a force-directed layout algorithm commonly used for visualizing graphs. By leveraging the computational power of GPUs and the parallel computing capabilities of CUDA, this project aims to significantly improve the performance of the algorithm.

Please note that this project is still in progress and further development is ongoing. The provided information in this readme represents the current state of the project.

## Features
- Accelerated Fruchterman Reingold algorithm using CUDA.
- Pybind as a wrapper for data manipulation and visualization.
- Efficient parallel computations on GPU to improve algorithm performance.
- Visualization capabilities for the generated graph layout.

## Requirements
To run this project, the following requirements must be met:
- CUDA-enabled GPU with compute capability 3.0 or higher.
- Python 3.x.
- Pybind.
- CUDA Toolkit.
- CUDA-compatible compiler.
- Additional dependencies as specified in the project.

## Installation
1. Clone or download the project repository from the project's GitHub page.
2. Install the required dependencies as mentioned in the Requirements section.
3. Build the CUDA kernel and associated code using the provided build script or instructions in the project's documentation.
4. Run the project by executing the main script or following the instructions in the project's documentation.

## Usage
To use the accelerated Fruchterman Reingold algorithm on GPU, follow these steps:
1. Prepare your input graph data in a compatible format (e.g., adjacency matrix or edge list).
2. Import the necessary modules and functions from the project.
3. Load the input graph data into the project.
4. Call the appropriate function(s) to run the Fruchterman Reingold algorithm on the GPU.
5. Retrieve the computed graph layout from the GPU.
6. Visualize the graph layout using the provided visualization tools or export the layout data for use in other applications.

## Contributing
Contributions to this project are welcome. If you would like to contribute, please follow these steps:
1. Fork the repository and create a new branch.
2. Make your changes and test them thoroughly.
3. Submit a pull request, clearly describing the changes and their purpose.

## License
MIT

## Contact
For any questions, issues, or suggestions related to this project, please contact the project owner via the provided contact information in the project's documentation or GitHub page.

## Disclaimer
This project is still in progress and may contain bugs or incomplete features. Use it at your own risk and take appropriate measures to ensure data integrity and system stability. The project owner and contributors are not responsible for any damages or losses incurred while using this project.
