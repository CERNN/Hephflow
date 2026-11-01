# Hephflow

[![License: GPL v2](https://img.shields.io/badge/License-GPLv2-blue.svg)](./LICENSE.txt)

Hephflow is a CUDA-accelerated, moments-based Lattice Boltzmann Method (LBM)
solver for three-dimensional computational fluid dynamics. It is the successor
to [MR-LBM](https://github.com/CERNN/MR-LBM) and builds on the earlier
[VISCOPLASTIC-LBM](https://github.com/CERNN/VISCOPLASTIC-LBM) codebase.

The project provides compile-time simulation cases for Newtonian and
non-Newtonian flows, scalar transport, thermal and multiphase models,
viscoelastic flows, curved boundaries, and immersed particles. D3Q19 and D3Q27
velocity sets are supported, and the domain can be decomposed across multiple
NVIDIA GPUs in the Z direction.

> [!NOTE]
> Hephflow is research software under active development. Case definitions and
> numerical models should be validated for the intended application before
> production use.

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configure a simulation](#configure-a-simulation)
- [Build and run](#build-and-run)
- [Outputs and visualization](#outputs-and-visualization)
- [Create a case](#create-a-case)
- [Tests and benchmarks](#tests-and-benchmarks)
- [Repository layout](#repository-layout)
- [Publications](#publications)
- [Contributing](#contributing)
- [License and support](#license-and-support)

## Requirements

- An NVIDIA CUDA-capable GPU
- A compatible NVIDIA driver and CUDA Toolkit (`nvcc` must be available)
- A C++17 host compiler supported by the installed CUDA Toolkit
- Bash on Linux, or Git Bash/MSYS2/Cygwin on Windows, for the provided build
  script
- [ParaView](https://www.paraview.org/) for viewing VTK output (optional)
- Python 3 for the regression-test and parameter-sweep tools (optional)

Check the CUDA installation before building:

```bash
nvidia-smi
nvcc --version
```

The CUDA Toolkit version reported by `nvcc` must be supported by the installed
driver.


1. **Install CUDA Toolkit**: Download from [Nvidia's website](https://developer.nvidia.com/cuda-toolkit)
2. **Set active case**: Edit `src/var.h` and change `#define BC_PROBLEM` to your desired path below `src/cases/` (e.g., `Newtonian/Laminar/001_lidDrivenCavity_3D`)
3. **Compile**: Run `sh compile.sh <VELOCITY_SET> <CASE_ID>` (e.g., `sh compile.sh D3Q19 001`)
   - The script auto-detects your GPU's compute capability via `nvidia-smi`
4. **Run**: Execute the compiled binary from the `bin/` directory
5. **Visualize**: Open generated `.vtk` files directly in ParaView (no post-processing needed for VTK output)  


The selected case is compiled into the executable. The repository currently
uses `000_benchmark` by default.

```bash
git clone https://github.com/CERNN/Hephflow.git
cd Hephflow/src
sh compile.sh D3Q19 000
../bin/000sim_D3Q19_sm80
```

Replace `sm80` with the compute capability detected for your GPU. On Windows,
the generated program has an `.exe` suffix.

To run another case:

1. Set `BC_PROBLEM` in [`src/var.h`](./src/var.h), for example:

   ```cpp
   #define BC_PROBLEM 001_lidDrivenCavity_3D
   ```

2. Check the velocity set in that case's `model.inc`.
3. Pass the same velocity set and a desired output prefix to `compile.sh`:

1. **Install CUDA Toolkit**:
   - Download the CUDA toolkit from [NVIDIA's official website](https://developer.nvidia.com/cuda-toolkit)
   - Follow the installation instructions for your operating system
   - Verify installation by running `nvcc --version` in your terminal

2. **Select Simulation Case**:
   - Edit `src/var.h` and modify `#define BC_PROBLEM` to select your case
   - Available cases are organized by physics under `src/cases/` (e.g., `Newtonian/Laminar/001_lidDrivenCavity_3D`)
   - Each case contains required `.inc` files defining the simulation parameters

3. **Compile the Project**:

   ```bash
   cd src
   sh compile.sh D3Q19 001
   ../bin/001sim_D3Q19_sm80
   ```

The second argument names the executable; it does not select `BC_PROBLEM`.

## Configure a simulation

Global compile-time settings live in [`src/var.h`](./src/var.h):

- `BC_PROBLEM` selects a directory under `src/cases/`.
- `N_GPUS` and `GPUS_TO_USE` select the CUDA devices.
- `GPU_INDEX` is retained for code paths that use a single device index.

For multi-GPU runs, `NZ` must be exactly divisible by `N_GPUS`. Peer access
must also be available between the selected devices.

Each case separates configuration by responsibility:

| File | Purpose |
|---|---|
| `model.inc` | Velocity set, collision operator, and optional physics models |
| `constants.inc` | Domain size, time steps, material properties, forces, and boundary modes |
| `output.inc` | Output directory, save intervals, VTK/binary output, and checkpoints |
| `bc_initialization.inc` | Assigns node types to the geometry |
| `bc_definition.inc` | Reconstructs hydrodynamic boundary conditions |
| `flow_initialization.inc` | Initializes density, velocity, and optional fields |

Additional files are enabled only by models that need them. Examples include
`g_bc_definition.inc` for scalar or thermal transport,
`phi_bc_definition.inc` for phase-field transport, `a*_bc_definition.inc` for
conformation-tensor components, `particleCreation.inc` for immersed particles,
and `force_field.inc` for spatially varying body forces.

### Precision and launch geometry

Floating-point precision is selected in
[`src/include/var_types.h`](./src/include/var_types.h). The default is
`SINGLE_PRECISION`; define only one of `SINGLE_PRECISION` or
`DOUBLE_PRECISION`.

CUDA block dimensions are set by `BLOCK_NX`, `BLOCK_NY`, and `BLOCK_NZ` in
[`src/include/memory_layout.h`](./src/include/memory_layout.h). Domain dimensions
must be at least as large as their corresponding block dimensions.

## Build and run

From `src/`, run:

```bash
sh compile.sh <D3Q19|D3Q27> <output-prefix>
```

The script:

- detects the first GPU's compute capability with `nvidia-smi` unless `CC` is
  already set;
- invokes `nvcc` with optimization, relocatable device code, and the required
  CUDA runtime libraries; and
- writes the executable to `bin/` as
  `<prefix>sim_<velocity-set>_sm<compute-capability>`.

Override auto-detection when compiling for another architecture:

```bash
CC=89 sh compile.sh D3Q19 001
```

The velocity-set argument must match the `D3Q19` or `D3Q27` definition in the
selected case's `model.inc`.

## Outputs and visualization

Outputs are written relative to the executable, under:

```text
bin/<PATH_FILES>/<ID_SIM>/
```

Both values and the save controls are defined in the selected case's
`output.inc`:

- `VTK_SAVE` writes legacy VTK files that can be opened directly in ParaView.
- `BIN_SAVE` writes raw binary field data for custom processing.
- `MACR_SAVE` controls periodic macroscopic-field saves.
- `ONLY_FINAL_MACRO` restricts macroscopic output to the final step.
- `CHECKPOINT_SAVE` and `LOAD_CHECKPOINT` control restart data.
- `MEAN_FLOW`, particle output, and force output are available when their
  corresponding models are enabled.

When a case defines physical VTK time, Hephflow also maintains a
`.vtk.series` file for loading the time sequence in ParaView.

Binary output can be converted to a legacy VTK structured-points file with the
dependency-free converter in `scripts/`. It discovers the saved fields and
steps from the output directory:

```bash
python scripts/bin_to_vtk.py "PATH_TO_OUTPUT/SIMULATION_ID"
```

The path may be an output directory or a path relative to `bin/`. Use
`--step N` or `--field NAME` to limit the conversion; pass either option more
than once to select multiple values. Run with `--help` for all options.

## Create a case

Start from the maintained template or from the closest validated case:

```bash
cd src/cases
cp -r _template 009_my_case
```

Then:

1. Configure the lattice and physics in `model.inc`.
2. Set the grid, run length, material properties, and boundary modes in
   `constants.inc`.
3. Implement geometry assignment in `bc_initialization.inc`.
4. Implement the required boundary reconstruction files.
5. Set the initial fields in `flow_initialization.inc`.
6. Configure the output in `output.inc`.
7. Set `BC_PROBLEM` in `src/var.h`, compile, and validate the case.

The `src/cases/` directory contains examples ranging from lid-driven cavities,
ducts, and Taylor-Green vortices to thermal convection, viscoelastic flows,
phase-field simulations, and immersed-particle problems.

### Voxel geometries

Voxel cases can load zero-based solid-node coordinates from CSV. Define
`VOXEL_FILENAME` in the case configuration and use the voxel boundary logic as
shown by `src/cases/Newtonian/Laminar/001_voxel`. The dependency-free
`scripts/generate_voxels.py` utility generates cubes, mounted cubes, parallel
plates, and spheres, and voxelizes closed binary or ASCII STL meshes. STL
meshes are fitted into the domain while preserving their aspect ratio. Run the
utility or a subcommand with `--help` for all parameters; for example:

```bash
python scripts/generate_voxels.py -o bin/sphere.csv sphere --domain 256 256 256 --diameter 32
python scripts/generate_voxels.py -o bin/model.csv stl model.stl --domain 256 256 256 --padding 2
```


## Repository layout

```text
Hephflow/
|-- src/
|   |-- cases/                     # Compile-time simulation cases
|   |-- collision_reconstruction/  # Hydrodynamic and scalar collision models
|   |-- fragments/                 # Reusable kernel and output fragments
|   |-- include/                   # Types, layouts, utilities, and configuration
|   |-- non_newtonian/             # Rheology implementations
|   |-- particles/                 # Immersed-boundary particle implementation
|   |-- compile.sh                 # Cross-platform Bash build script
|   |-- main.cu                    # Program entry point and time loop
|   `-- mlbm.cu                    # Main LBM kernels
|-- bin/                           # Executables and simulation output
|-- tests/                         # Regression-test runner and standalone tests
|-- scripts/                       # Geometry, post-processing, and performance tools
|-- docs/                          # Technical notes and reports
|-- sweep.py                       # Parameter-sweep runner
`-- Doxyfile                       # API documentation configuration
```
## Performance Benchmarks

The following table summarizes performance benchmarks on various NVIDIA GPUs. Performance is measured in MLUPs (Million Lattice Updates Per Second) using the D3Q19 lattice with FP32 precision.

**Note**: Performance may vary due to GPU frequency, thermal throttling, and system configuration.

### Desktop GPUs (Ada & Ampere Architecture)

| GPU | Compute Capability | Frequency | Memory | Block Size | MLUPs | Notes |
|-----|---|---|---|---|---|---|
| RTX 5090 | 120 | 2.8 GHz | 1.7 GHz | 8×8×8 | 13075 | Compute-limited (86%) |
| RTX 4090 OC | 89 | 3.0 GHz | 1.5 GHz | 8×8×8 | 9075 | Overclocked |
| RTX 4090 | 89 | 2.8 GHz | 1.3 GHz | 8×8×8 | 7899 | Stock |
| RTX 4060 | 89 | 2.8 GHz | 2.1 GHz | 8×8×8 | 2167 | - |
| RTX 4060 | 89 | 2.8 GHz | 2.1 GHz | 8×8×4 | 1932 | Reduced block |
| RTX 3060 OC | 86 | 2.0 GHz | 2.0 GHz | 8×8×8 | 3083 | Overclocked |
| RTX 3060 | 86 | 1.8 GHz | 1.8 GHz | 8×8×8 | 2755 | Stock |

### Datacenter GPUs

| GPU | Compute Capability | Frequency | Memory | Block Size | MLUPs | Notes |
|-----|---|---|---|---|---|---|
| A100 | 80 | 1.3 GHz | 1.5 GHz | 16×8×8 | 10243 | Dynamic shared mem |
| A100 | 80 | 1.3 GHz | 1.5 GHz | 8×8×8 | 11390 |  |

### Legacy GPUs (Maxwell & Kepler Architecture)

| GPU | Compute Capability | Frequency | Memory | Block Size | MLUPs | Notes |
|-----|---|---|---|---|---|---|
| RTX 2060 | 75 | 1.9 GHz | 1.7 GHz | 8×8×8 | 2397 | - |
| GTX 1660 | 75 | 1.9 GHz | 2.0 GHz | 8×8×8 | 1323 | - |
| GTX 1660 | 75 | 1.9 GHz | 2.0 GHz | 8×8×4 | 1248 | Reduced block |
| GTX 1660 | 75 | 1.9 GHz | 2.0 GHz | 16×4×4 | 1213 | Reduced block |
| T600 | 75 | 1.5 GHz | 1.25 GHz | 8×8×8 | 881 | Memory-limited (84%) |
| K20x | 35 | 0.7 GHz | 1.3 GHz | 8×8×4 | 730 | Memory-limited (47%) |
| K20x | 35 | 0.7 GHz | 1.3 GHz | 8×8×8 | 670 | Memory-limited (40%) |
| K80 | 35 | 0.8 GHz | 1.2 GHz | 8×8×8 | 1142 | Single GPU |

### Notes

- **MLUPs**: Million Lattice Updates Per Second
- **Dynamic Allocation**: The "D" designation indicates dynamic allocation of shared memory (48 KB per block)
- Performance may be affected by system thermal conditions and power delivery limitations

## Publications

- [DOI: 10.1063/5.0209802](https://doi.org/10.1063/5.0209802)
- [DOI: 10.1016/j.jnnfm.2024.105198](https://doi.org/10.1016/j.jnnfm.2024.105198)
- [ResearchGate: Evaluating the Impact of Boundary Conditions on MR-LBM](https://www.researchgate.net/publication/378070516_Evaluating_the_Impact_of_Boundary_Conditions_on_the_MR-LBM)
- [DOI: 10.1016/j.jnnfm.2023.105030](https://doi.org/10.1016/j.jnnfm.2023.105030)
- [DOI: 10.1002/fld.5185](https://doi.org/10.1002/fld.5185)

## Previous Version: MR-LBM

Hephflow is a direct successor to the **MR-LBM** project, incorporating improved numerical schemes and expanded capabilities.

If you're interested in the original MR-LBM implementation, visit: [MR-LBM on GitHub](https://github.com/CERNN/MR-LBM)

## Contributing

Bug reports, feature proposals, numerical validation cases, and code
contributions are welcome. Read [`CONTRIBUTING.md`](./CONTRIBUTING.md) before
opening a pull request. Planned work and known issues are tracked in
[`TODO.todo`](./TODO.todo).

## License and support

Hephflow is distributed under the
[GNU General Public License, version 2](./LICENSE.txt).

- Issues: [github.com/CERNN/Hephflow/issues](https://github.com/CERNN/Hephflow/issues)
- Contact: e.marcoferrari@utfpr.edu.br
