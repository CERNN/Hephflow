# Hephflow

[![License: GPL v2](https://img.shields.io/badge/License-GPLv2-blue.svg)](./LICENSE.txt)

This repository contains **Hephflow**, a moment-based implementation of the Lattice Boltzmann Method (LBM) for GPU acceleration using CUDA. Hephflow is the successor to the previous project **MR-LBM**, and builds upon its concepts and codebase, introducing new features and improvements for efficient computational fluid dynamics simulations on GPU hardware.

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Installation Guide](#installation-guide)
- [Simulation](#simulation)
- [Post Processing](#post-processing)
- [File Structure](#file-structure)
- [Creating a Boundary Case](#creating-a-boundary-case)
- [Using Voxels for Immersed Bodies](#using-voxels-for-immersed-bodies)
- [Performance Benchmarks](#performance-benchmarks)
- [Gallery](#gallery)
- [Publications](#publications)
- [Previous Version](#previous-version-mr-lbm)
- [License](#license)
- [Contact](#contact)

## Project Overview

**Hephflow** implements the moment-based Lattice Boltzmann Method, where the collision operator is formulated across moments from the 0th to the 2nd order (or higher). This approach provides improved stability and accuracy for a wide range of fluid dynamics problems, including Newtonian, non-Newtonian, and thermal flows.

This project is intended primarily as a proof of concept, with many features still under development and optimization.

Hephflow builds on the codebase of the previous project [VISCOPLASTIC-LBM](https://github.com/CERNN/VISCOPLASTIC-LBM) and shares similar licensing terms and design philosophy.

## Quick Start

1. **Install CUDA Toolkit**: Download from [Nvidia's website](https://developer.nvidia.com/cuda-toolkit)
2. **Modify compile script** (if needed): Edit `src/compile.sh` or `src/compile_linux.sh` based on your GPU's compute capability
3. **Compile**: Run `sh compile.sh <VELOCITY_SET> <CASE_ID>` (e.g., `sh compile.sh D3Q19 000`)
4. **Run**: Execute the compiled binary from the build directory
5. **Post-process**: Use the Python scripts in the `post/` folder to convert binary output to Paraview-compatible formats  


## Installation Guide

### System Requirements

To compile and run Hephflow, you will need:

- **C++ Compiler** (MSVC on Windows, GCC on Linux)
- **NVIDIA GPU** with compute capability 3.5 or higher
- **NVIDIA GPU Drivers** (latest version recommended)
- **CUDA Toolkit** (includes both GPU drivers and necessary CUDA libraries)

**Note**: Hephflow is designed to run on a single GPU. Multi-GPU setups are not currently supported.

### Installation Steps

1. **Install CUDA Toolkit**:
   - Download the CUDA toolkit from [NVIDIA's official website](https://developer.nvidia.com/cuda-toolkit)
   - Follow the installation instructions for your operating system
   - Verify installation by running `nvcc --version` in your terminal

2. **Configure Compilation Script**:
   - On Windows: Use `src/compile.sh` (automatically detects GPU compute capability via nvidia-smi)
   - On Linux: Use `src/compile_linux.sh` and set the `CC` variable to your GPU's compute capability
   - Refer to the script comments for guidance on modifying parameters

3. **Compile the Project**:
   ```bash
   cd src
   sh compile.sh <VELOCITY_SET> <CASE_ID>
   ```
   Example: `sh compile.sh D3Q19 000`

4. **Verify Compilation**:
   - Check for the compiled executable in the build directory
   - Run a test case to ensure proper functionality

## Simulation

### Output Files

The simulation generates:
- **Binary output files**: Contain macroscopic quantities such as density, velocity, and pressure fields
- **Information file**: Describes simulation parameters (lattice size, relaxation time, velocity set, collision model, etc.)

To convert these binary files into formats compatible with visualization software (e.g., ParaView), use the provided Python post-processing tools. 

## Post Processing

### Requirements

To process the output binary files, you will need:
- Python 3.6 or higher
- Dependencies: `glob`, `numpy`, `os`, `pyevtk`, `matplotlib`

### Converting Binary Output to Paraview Format

Use the Python post-processing script to convert binary files to `.vtr` format (compatible with ParaView):

```bash
cd post
python bin2VTK.py "PATH_TO_OUTPUT/SIMULATION_ID"
```

This will generate VTK files that can be imported into ParaView for visualization and analysis.

## File Structure

The following table provides an overview of key files and directories in the project:

| # | File/Folder | Description |
|---|---|---|
| 1 | `main.cu` | Main application entry point |
| 2 | `mlbm.cu`/`mlbm.cuh` | Core kernel implementing streaming-collision operations |
| 3 | `var.h` | Global simulation parameters and configuration |
| 4 | `compile.sh` | Compilation script (edit for correct CUDA version and compute capability) |
| 5 | `definitions.h` | LBM constants and macro definitions |
| 6 | `arrayIndex.h` | Index calculation functions for moment ordering |
| 7 | `auxFunctions.cu`/`auxFunctions.cuh` | Auxiliary GPU functions |
| 8 | `cases/` | Folder containing simulation case definitions |
| 9 | `cases/<CASE_ID>/bc_definition.inc` | Mathematical moment equations for boundary conditions |
| 10 | `cases/<CASE_ID>/bc_initialization.inc` | Boundary condition flag definitions |
| 11 | `cases/<CASE_ID>/constants.inc` | Simulation parameters (mesh size, velocity, relaxation times) |
| 12 | `cases/<CASE_ID>/flow_initialization.inc` | Initial flow field setup |
| 13 | `cases/<CASE_ID>/model.inc` | Model selection (velocity set, collision operator) |
| 14 | `cases/<CASE_ID>/output.inc` | Output and data export configuration |
| 15 | `colrec/` | Collision and reconstruction implementations for various lattice models |
| 16 | `includeFiles/popSave` | Kernel for loading populations from global memory |
| 17 | `includeFiles/popLoad` | Kernel for saving populations to global memory |
| 18 | `includeFiles/interface` | Boundary condition interface definitions (wall, periodic) |
| 19 | `checkpoint.cu`/`checkpoint.cuh` | Checkpoint/restart functionality |
| 20 | `errorDef.h` | Error handling and validation functions |
| 21 | `globalFunctions.cu`/`globalFunctions.h` | Global utility and indexing functions |
| 22 | `globalStructs.h` | CUDA device and host structures |
| 23 | `lbmInitialization.cu`/`lbmInitialization.cuh` | Field and lattice initialization |
| 24 | `nnf.h` | Non-Newtonian fluid model definitions |
| 25 | `nodeTypeMap.h` | Node type mapping for boundary conditions |
| 26 | `reduction.cu`/`reduction.cuh` | Parallel reduction kernels |
| 27 | `saveData.cu`/`saveData.cuh` | Data serialization and output functions |
| 28 | `post/` | Post-processing utilities for result visualization |

## Creating a Boundary Case

To define a new simulation case, create the following files in the `src/cases/` directory:

### Required Files

1. **`bc_definition`**: Defines the mathematical moment equations (0th to 2nd order) and boundary condition implementations
2. **`bc_initialization`**: Sets boundary condition flags for each node type
3. **`constants`**: Specifies simulation parameters:
   - Mesh dimensions (Nx, Ny, Nz)
   - Flow velocity and Mach number
   - Relaxation times (tau values)
4. **`flow_initialization`**: Defines initial velocity and density fields
5. **`model`**: Specifies model parameters:
   - Velocity set (D2Q9, D3Q19, D3Q27, etc.)
   - Collision operator (MRT, CUMULATIVE, etc.)
6. **`output`**: Configures data export:
   - Output frequency
   - Variables to save
   - File paths

### Optional Files

For advanced simulations, additional files may be required:
- **Thermal models**: Temperature and heat flux moment definitions
- **Non-Newtonian models**: Viscosity and stress tensor calculations
- **Multiphase models**: Phase field and interfacial tension parameters

## Using Voxels for Immersed Bodies

Voxelization enables efficient representation of complex immersed solid bodies in the lattice. To use voxels:

### Setup Steps

1. **Create a coordinate file**:
   - Generate a CSV file listing all solid node coordinates
   - Format: One coordinate per line (e.g., `x,y,z`)

2. **Configure the simulation case**:
   - Add `#define VOXEL_FILENAME "path/to/your/voxel/file.csv"` in the `constants.inc` file
   - Specify the path to your voxel coordinate file

3. **Enable voxel boundary conditions**:
   - Include `#ifdef VOXEL_FILENAME` guard in your `bc_definition.inc` file
   - Add voxel-specific boundary condition logic within the guard
   - This enables proper handling of voxel nodes

### Support Files

Support files for voxel generation are provided in `aSupportFiles/`:
- `VOXELISE.m`: MATLAB function for voxel mesh generation
- `geometry.py`: Python utilities for geometry processing

## Performance Benchmarks

The following table summarizes performance benchmarks on various NVIDIA GPUs. Performance is measured in MLUPs (Million Lattice Updates Per Second) using the D3Q19 lattice with FP32 precision.

**Note**: Performance may vary due to GPU frequency, thermal throttling, and system configuration.

### Desktop GPUs (Ada & Ampere Architecture)

| GPU | Compute Capability | Frequency | Memory | Block Size | MLUPs | Notes |
|-----|---|---|---|---|---|---|
| RTX 4090 OC | 89 | 3.0 GHz | 1.5 GHz | 8×8×8 | 9,075 | Overclocked |
| RTX 4090 | 89 | 2.8 GHz | 1.3 GHz | 8×8×8 | 7,899 | Stock |
| RTX 4060 | 89 | 2.8 GHz | 2.1 GHz | 8×8×8 | 2,167 | - |
| RTX 4060 | 89 | 2.8 GHz | 2.1 GHz | 8×8×4 | 1,932 | Reduced block |
| RTX 3060 OC | 86 | 2.0 GHz | 2.0 GHz | 8×8×8 | 3,083 | Overclocked |
| RTX 3060 | 86 | 1.8 GHz | 1.8 GHz | 8×8×8 | 2,755 | Stock |

### Datacenter GPUs

| GPU | Compute Capability | Frequency | Memory | Block Size | MLUPs | Notes |
|-----|---|---|---|---|---|---|
| A100 | 80 | 1.3 GHz | 1.5 GHz | 16×8×8 | 10,243 | Dynamic shared mem |
| A100 | 80 | 1.3 GHz | 1.5 GHz | 8×8×8 | 11,390 | Peak performance |

### Legacy GPUs (Maxwell & Kepler Architecture)

| GPU | Compute Capability | Frequency | Memory | Block Size | MLUPs | Notes |
|-----|---|---|---|---|---|---|
| RTX 2060 | 75 | 1.9 GHz | 1.7 GHz | 8×8×8 | 2,397 | - |
| GTX 1660 | 75 | 1.9 GHz | 2.0 GHz | 8×8×8 | 1,323 | - |
| GTX 1660 | 75 | 1.9 GHz | 2.0 GHz | 8×8×4 | 1,248 | Reduced block |
| GTX 1660 | 75 | 1.9 GHz | 2.0 GHz | 16×4×4 | 1,213 | Reduced block |
| T600 | 75 | 1.5 GHz | 1.25 GHz | 8×8×8 | 881 | Memory-limited (84%) |
| K20x | 35 | 0.7 GHz | 1.3 GHz | 8×8×4 | 730 | Memory-limited (47%) |
| K20x | 35 | 0.7 GHz | 1.3 GHz | 8×8×8 | 670 | Memory-limited (40%) |
| K80 | 35 | 0.8 GHz | 1.2 GHz | 8×8×8 | 1,142 | Single GPU |

### Notes

- **MLUPs**: Million Lattice Updates Per Second
- **Dynamic Allocation**: The "D" designation indicates dynamic allocation of shared memory (48 KB per block)
- Performance may be affected by system thermal conditions and power delivery limitations

## Gallery

*In progress*

## Publications

- [DOI: 10.1063/5.0209802](https://doi.org/10.1063/5.0209802)
- [DOI: 10.1016/j.jnnfm.2024.105198](https://doi.org/10.1016/j.jnnfm.2024.105198)
- [ResearchGate: Evaluating the Impact of Boundary Conditions on MR-LBM](https://www.researchgate.net/publication/378070516_Evaluating_the_Impact_of_Boundary_Conditions_on_the_MR-LBM)
- [DOI: 10.1016/j.jnnfm.2023.105030](https://doi.org/10.1016/j.jnnfm.2023.105030)
- [DOI: 10.1002/fld.5185](https://doi.org/10.1002/fld.5185)

## Previous Version: MR-LBM

Hephflow is a direct successor to the **MR-LBM** project, incorporating improved numerical schemes and expanded capabilities.

If you're interested in the original MR-LBM implementation, visit: [MR-LBM on GitHub](https://github.com/CERNN/MR-LBM)

## Development Notes

For a complete list of planned features, ongoing improvements, and known issues, refer to the [TODO.todo](./TODO.todo) file. This file tracks performance optimizations, feature enhancements, and potential extensions to the current implementation.

## License

Hephflow is distributed under the [GNU General Public License v2 (GPLv2)](./LICENSE.txt). This ensures that the software remains free and open-source.

## Contact & Support

For bug reports, feature requests, or other issues:

1. **GitHub Issues**: Please use the [GitHub issue tracker](https://github.com/CERNN/Hephflow/issues) for visibility and community discussion
2. **Email**: You can also reach the maintainers at:
   - e.marcoferrari@utfpr.edu.br

Thank you for your interest in Hephflow!
