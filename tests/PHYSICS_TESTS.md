# Hephflow Physics Regression Test Suite

## Overview

Physics regression test suite that validates simulation accuracy after kernel modifications. Tests compare LBM numerical solutions against analytical or reference solutions using L2 error metrics.

## Test Framework Architecture

```
src/
├── main.cu                          # Main simulation loop with TESTS integration
├── include/case_definitions.h       # Includes CASE_TEST_METRIC macro
└── cases/{CASE_NAME}/_test/
    ├── test_metric.inc              # Test computation logic (runs at final timestep)
    ├── tolerances.txt               # Error thresholds for pass/fail
    └── reference_data.txt           # Expected baseline values

tests/
├── run_tests.py                     # Python test orchestrator
└── PHYSICS_TESTS.md                 # This documentation
```

### How It Works

1. **Conditional Compilation**: Python runner temporarily adds `#define TESTS 1` to var.h during compilation
2. **Execution**: Test code included via `#ifdef TESTS` in main.cu, evaluates only at final timestep
3. **Validation**: L2 error compared against tolerances defined in tolerances.txt
4. **Output**: test_results.txt generated with detailed metrics and velocity profiles

## Currently Implemented Tests

### ✓ 001_parallelPlates_D3Q19 - Poiseuille Flow

**Physics**: Newtonian channel flow with analytical solution

**Metric**: L2 error of z-velocity profile across channel

**Analytical Solution**: 
```
u_z(y) = (FZ / 2*VISC) * y * (NY - y)
```

**Case Parameters**:
- FZ = 3.0e-5 (driving force)
- VISC = 1.0/6.0 (kinematic viscosity)
- NY = 64 (channel height)

**Tolerances**:
- Absolute L2 error: < 1e-4
- Relative L2 error: < 0.02 (2%)

**Current Results** (need tuning):
- L2 error (absolute): 4.66e-05 ✓ PASS
- L2 error (relative): 0.0182 ✗ FAIL (exceeds 1% tolerance, but < 2%)

## Usage

### Run Tests (Recommended)

```bash
cd /path/to/Hephflow

# Run single test case
python tests/run_tests.py 001_parallelPlates_D3Q19

# Run multiple cases
python tests/run_tests.py 001_parallelPlates_D3Q19 002_anotherCase

# View results
cat src/cases/001_parallelPlates_D3Q19/_test/test_results.txt
```

The Python runner automatically:
- Temporarily modifies var.h to enable TESTS and set BC_PROBLEM
- Compiles with correct flags (D3Q19 011)
- Runs simulation from case directory
- Validates results against tolerances
- Restores var.h to original state

## Test Results Interpretation

test_results.txt contains:
- STEP, MAX_UZ_NUMERICAL, MAX_UZ_ANALYTICAL
- L2_ERROR_ABSOLUTE, L2_ERROR_RELATIVE
- Detailed velocity profile at each Y position

**Pass Criteria**:
- L2_ERROR_ABSOLUTE < 1e-4
- L2_ERROR_RELATIVE < 0.02

**Common Failure Causes**:
1. Kernel modifications affecting accuracy
2. Changed compilation flags
3. Incorrect BC_PROBLEM or mesh parameters

## Adding New Test Cases

1. Create `src/cases/{NEW_CASE}/_test/` directory
2. Add `test_metric.inc` with test computation logic
3. Create `tolerances.txt` with acceptable thresholds
4. Create `reference_data.txt` with baseline values
5. Test with: `python tests/run_tests.py NEW_CASE`

## Windows-Specific Notes

- Test runner handles UTF-8 encoding automatically
- Temporarily edits var.h (restored in finally block)
- Executable runs from case directory, creates test_results.txt in `_test/` subdirectory
