# Simulation case organization

Cases are grouped by their primary physical model. Set `BC_PROBLEM` to the
complete path below this directory, without quotes:

```cpp
#define BC_PROBLEM Newtonian/Laminar/001_taylorGreen
```

The current categories are:

- `Benchmarks/`: performance and IBM benchmark cases.
- `Newtonian/Laminar/`: laminar and general single-phase Newtonian flows.
- `Newtonian/Turbulent/`: turbulent Newtonian flows.
- `Viscoplastic/Bingham/`: Bingham-fluid cases.
- `Viscoplastic/HerschelBulkley/`: Herschel-Bulkley cases.
- `Viscoelastic/`: viscoelastic cases, including an `ImmersedBoundaries/`
  subgroup for coupled configurations.
- `ScalarTransport/`: transported-scalar verification cases.
- `Thermal/`: thermal flows, including a `Viscoplastic/` subgroup.
- `ImmersedBoundaries/`: fixed geometry, Couette, settling, and interaction
  cases.
- `Multiphase/`: phase-field cases, including a `ComplexRheology/` subgroup.
- `Templates/`: the reusable case template.
- `Experimental/`: incomplete or exploratory case directories.

Case directory names retain their numeric prefixes for continuity with output
identifiers and existing documentation. The category path is part of
`BC_PROBLEM`; for example:

```cpp
#define BC_PROBLEM Viscoplastic/Bingham/002_viscoplasticJet
#define BC_PROBLEM Thermal/005_rayleighBenardCube
#define BC_PROBLEM Multiphase/ComplexRheology/007_twoLayerChannel_NV
```
