#!/usr/bin/env python3
"""
Test runner for Hephflow physics regression tests
Orchestrates compilation, execution, and result validation

Usage:
    # Run all configured tests in TEST_SUITE
    python tests/run_tests.py
    
    # Run specific test case(s)
    python tests/run_tests.py 001_parallelPlates_D3Q19
    python tests/run_tests.py 001_parallelPlates_D3Q19 001_parallelPlates_D3Q27
"""

import sys
import subprocess
import re
from pathlib import Path
from typing import Dict, Tuple

# Fix Unicode encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Test suite configuration: maps case names to their compilation parameters
TEST_SUITE = {
    '001_parallelPlates_D3Q19': {
        'velocity_set': 'D3Q19',
        'case_number': '011',
        'block_size': (8, 8, 8)  # (BLOCK_NX, BLOCK_NY, BLOCK_NZ)
    },
    '001_parallelPlates_D3Q27': {
        'velocity_set': 'D3Q27',
        'case_number': '011',
        'block_size': (8, 8, 4)  # Smaller blocks for D3Q27 to fit shared memory
    },
    # Add more test cases here as needed:
    # '001_lidDrivenCavity_2D': {
    #     'velocity_set': 'D3Q19',
    #     'case_number': '001',
    #     'block_size': (8, 8, 8)
    # },
}

class TestRunner:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.src_dir = repo_root / "src"
        self.cases_dir = self.src_dir / "cases"
        self.bin_dir = repo_root / "bin"
        
    def run_test_case(self, case_name: str) -> Tuple[bool, Dict]:
        """
        Run a single test case
        Returns: (passed, results_dict)
        """
        case_dir = self.cases_dir / case_name
        test_dir = case_dir / "_test"
        
        if not test_dir.exists():
            print(f"⊘ Skipping {case_name} - no _test folder")
            return None, {}
        
        print(f"\n{'='*70}", flush=True)
        print(f"Testing: {case_name}", flush=True)
        print(f"{'='*70}", flush=True)
        
        results = {
            'case': case_name,
            'status': 'FAILED',
            'passed': False,
            'errors': []
        }
        
        # Step 1: Load test configuration
        print(f"[1/4] Loading test configuration...", flush=True)
        tolerances = self._load_tolerances(test_dir / "tolerances.txt")
        reference = self._load_reference(test_dir / "reference_data.txt")
        
        if not tolerances or not reference:
            results['errors'].append("Failed to load test configuration")
            return False, results
        
        # Step 2: Compile with TESTS enabled
        print(f"[2/4] Compiling {case_name} with TESTS enabled...", flush=True)
        if not self._compile_case(case_name):
            results['errors'].append("Compilation failed")
            return False, results
        
        # Step 3: Run simulation
        print(f"[3/4] Running simulation...", flush=True)
        if not self._run_simulation(case_dir, case_name):
            results['errors'].append("Simulation failed or timed out")
            return False, results
        
        # Step 4: Parse results and compare
        print(f"[4/4] Validating results...", flush=True)
        # Try both locations for test_results.txt
        test_result_paths = [
            case_dir / "test_results.txt",
            case_dir / "_test" / "test_results.txt"
        ]
        
        test_results = None
        for path in test_result_paths:
            test_results = self._parse_test_results(path)
            if test_results:
                print(f"  Found test results at: {path}")
                break
        
        if not test_results:
            results['errors'].append("Failed to parse test results")
            return False, results
        
        # Compare with tolerances
        passed, comparison = self._validate_results(
            test_results, 
            reference, 
            tolerances
        )
        
        results.update(comparison)
        results['passed'] = passed
        results['status'] = 'PASSED' if passed else 'FAILED'
        
        return passed, results
    
    def _load_tolerances(self, tol_file: Path) -> Dict:
        """Parse tolerances.txt file"""
        tolerances = {}
        try:
            with open(tol_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        try:
                            # Try to convert to float
                            tolerances[key] = float(value)
                        except ValueError:
                            # Keep as string
                            tolerances[key] = value
            
            return tolerances
        except FileNotFoundError:
            print(f"✗ Tolerances file not found: {tol_file}")
            return None
    
    def _load_reference(self, ref_file: Path) -> Dict:
        """Parse reference_data.txt file"""
        reference = {}
        try:
            with open(ref_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if ',' in line:
                        key, value = line.split(',', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        try:
                            reference[key] = float(value)
                        except ValueError:
                            reference[key] = value
            
            return reference
        except FileNotFoundError:
            print(f"✗ Reference file not found: {ref_file}")
            return None
    
    def _compile_case(self, case_name: str) -> bool:
        """Compile case with TESTS define"""
        # Get compilation configuration for this case
        if case_name not in TEST_SUITE:
            print(f"  [ERROR] Case {case_name} not in TEST_SUITE configuration")
            return False
        
        config = TEST_SUITE[case_name]
        velocity_set = config['velocity_set']
        case_number = config['case_number']
        block_nx, block_ny, block_nz = config['block_size']
        
        var_h_path = self.src_dir / "var.h"
        memory_layout_path = self.src_dir / "include" / "memory_layout.h"
        
        try:
            # Read original files
            with open(var_h_path, 'r') as f:
                var_h_original = f.read()
            with open(memory_layout_path, 'r') as f:
                memory_layout_original = f.read()
            
            # Update BC_PROBLEM definition
            var_h_modified = re.sub(
                r'#define BC_PROBLEM \w+',
                f'#define BC_PROBLEM {case_name}',
                var_h_original
            )
            
            # Add TESTS define right after BC_PROBLEM
            var_h_modified = re.sub(
                f'(#define BC_PROBLEM {case_name}\n)',
                f'#define BC_PROBLEM {case_name}\n#define TESTS 1\n',
                var_h_modified
            )
            
            # Write modified var.h
            with open(var_h_path, 'w') as f:
                f.write(var_h_modified)
            
            # Modify memory_layout.h with block dimensions
            memory_layout_modified = re.sub(
                r'#define BLOCK_NX \d+',
                f'#define BLOCK_NX {block_nx}',
                memory_layout_original
            )
            memory_layout_modified = re.sub(
                r'#define BLOCK_NY \d+',
                f'#define BLOCK_NY {block_ny}',
                memory_layout_modified
            )
            memory_layout_modified = re.sub(
                r'#define BLOCK_NZ \d+',
                f'#define BLOCK_NZ {block_nz}',
                memory_layout_modified
            )
            
            with open(memory_layout_path, 'w') as f:
                f.write(memory_layout_modified)
            
            print(f"  Updated BC_PROBLEM to {case_name} and enabled TESTS in var.h", flush=True)
            print(f"  Updated block size to {block_nx}x{block_ny}x{block_nz} in memory_layout.h", flush=True)
            print(f"  Compiling with {velocity_set} {case_number}...", flush=True)
            
            # Run compile script with case-specific parameters
            result = subprocess.run(
                ["bash", str(self.src_dir / "compile.sh"), velocity_set, case_number],
                cwd=str(self.src_dir),
                capture_output=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"  [FAIL] Compilation failed with exit code {result.returncode}")
                stderr_output = result.stderr.decode('utf-8', errors='ignore')
                stdout_output = result.stdout.decode('utf-8', errors='ignore')
                
                # Show last 20 lines of output for debugging
                print(f"\n  Last lines of compilation output:")
                all_output = stdout_output + stderr_output
                for line in all_output.split('\n')[-20:]:
                    if line.strip():
                        print(f"    {line}")
                return False
            else:
                # Print ALL compilation output to see debug messages
                output = result.stdout.decode('utf-8', errors='ignore')
                if output:
                    for line in output.split('\n'):
                        if "Test" in line or "Block" in line or "Max registers" in line:
                            print(f"  {line}")
            
            print(f"  [PASS] Compilation successful", flush=True)
            return True
            
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] Compilation timed out")
            return False
        except Exception as e:
            print(f"  [ERROR] Compilation error: {e}")
            return False
        finally:
            # Restore var.h to original state
            try:
                with open(var_h_path, 'r') as f:
                    var_h_content = f.read()
                
                # Remove the TESTS define line
                var_h_restored = re.sub(
                    r'#define TESTS 1\n',
                    '',
                    var_h_content
                )
                
                with open(var_h_path, 'w') as f:
                    f.write(var_h_restored)
                
                print(f"  Restored var.h (removed TESTS define)")
            except Exception as e:
                print(f"  [WARNING] Failed to restore var.h: {e}")
            
            # Restore memory_layout.h to original state
            try:
                with open(memory_layout_path, 'w') as f:
                    f.write(memory_layout_original)
                
                print(f"  Restored memory_layout.h to original block size")
            except Exception as e:
                print(f"  [WARNING] Failed to restore memory_layout.h: {e}")
    
    def _run_simulation(self, case_dir: Path, case_name: str) -> bool:
        """Run simulation executable"""
        try:
            # Determine executable name based on case configuration
            config = TEST_SUITE[case_name]
            velocity_set = config['velocity_set']
            case_number = config['case_number']
            sim_executable = self.bin_dir / f"{case_number}sim_{velocity_set}_sm86.exe"
            
            if not sim_executable.exists():
                print(f"  [ERROR] Executable not found: {sim_executable}")
                return False
            
            result = subprocess.run(
                [str(sim_executable)],
                cwd=str(case_dir),
                capture_output=True,
                timeout=600  # 10 minute timeout
            )
            
            # Check for test output in stdout
            output = result.stdout.decode('utf-8', errors='ignore')
            stderr = result.stderr.decode('utf-8', errors='ignore')
            
            # Look for test output
            has_test_output = False
            for line in output.split('\n'):
                if "[TEST]" in line:
                    print(f"    TEST OUTPUT: {line}")
                    has_test_output = True
            
            if not has_test_output:
                print(f"  No [TEST] output found in simulation")
                # Print last few lines of output for debugging
                output_lines = output.split('\n')[-10:]
                for line in output_lines:
                    if line.strip():
                        print(f"    {line}")
            
            if result.returncode != 0:
                print(f"  [FAIL] Simulation failed (exit code: {result.returncode})")
                if stderr:
                    print(f"  stderr: {stderr[-500:]}")
                return False
            
            print(f"  [PASS] Simulation completed", flush=True)
            return True
            
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] Simulation timed out (>10 min)")
            return False
        except Exception as e:
            print(f"  [ERROR] Simulation error: {e}")
            return False
    
    def _parse_test_results(self, results_file: Path) -> Dict:
        """Parse test_results.txt file generated by test_metric.inc"""
        results = {}
        try:
            with open(results_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if ',' in line and not any(c.isalpha() for c in line.split(',')[0]):
                        # Skip data rows (Y_INDEX,...)
                        continue
                    
                    if ',' in line:
                        key, value = line.split(',', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        try:
                            results[key] = float(value)
                        except ValueError:
                            results[key] = value
            
            return results if results else None
            
        except FileNotFoundError:
            print(f"  [ERROR] Test results file not found: {results_file}")
            return None
    
    def _validate_results(self, test_results: Dict, reference: Dict, 
                         tolerances: Dict) -> Tuple[bool, Dict]:
        """Compare test results with reference and tolerances"""
        comparison = {
            'test_metrics': test_results,
            'reference_metrics': reference,
            'violations': []
        }
        
        # Check L2 error absolute
        if 'L2_ERROR_ABSOLUTE' in test_results and 'EXPECTED_L2_ERROR_ABSOLUTE' in reference:
            measured = test_results['L2_ERROR_ABSOLUTE']
            expected = reference['EXPECTED_L2_ERROR_ABSOLUTE']
            tolerance = tolerances.get('L2_ERROR_ABSOLUTE_TOL', expected * 0.05)
            
            if measured > tolerance:
                comparison['violations'].append(
                    f"L2_ERROR_ABSOLUTE: {measured:.2e} > tolerance {tolerance:.2e}"
                )
                print(f"  ✗ L2 Error (absolute): {measured:.2e} (tolerance: {tolerance:.2e})")
            else:
                print(f"  ✓ L2 Error (absolute): {measured:.2e} (tolerance: {tolerance:.2e})")
        
        passed = len(comparison['violations']) == 0
        return passed, comparison
    
    def run_all_tests(self, test_cases: list = None) -> Tuple[int, int]:
        """Run all specified test cases"""
        if test_cases is None:
            test_cases = ['001_parallelPlates_D3Q19']
        
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        
        results_summary = []
        
        for case in test_cases:
            result = self.run_test_case(case)
            if result[0] is None:
                skipped_count += 1
            elif result[0]:
                passed_count += 1
                results_summary.append(result[1])
            else:
                failed_count += 1
                results_summary.append(result[1])
        
        # Print summary
        print(f"\n\n{'='*70}", flush=True)
        print(f"TEST SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Passed:  {passed_count}")
        print(f"Failed:  {failed_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Total:   {passed_count + failed_count + skipped_count}")
        
        for result in results_summary:
            status_symbol = "✓" if result['passed'] else "✗"
            print(f"\n{status_symbol} {result['case']}: {result['status']}")
            if result['errors']:
                for error in result['errors']:
                    print(f"    - {error}")
            if 'violations' in result and result['violations']:
                for violation in result['violations']:
                    print(f"    - {violation}")
        
        print(f"\n{'='*70}\n")
        
        return passed_count, failed_count

def main():
    # Detect repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent if script_dir.name == 'tests' else script_dir
    
    print(f"Repository root: {repo_root}", flush=True)
    print(f"Test directory: {script_dir}\n", flush=True)
    
    runner = TestRunner(repo_root)
    
    # Run tests - use command line args or run all configured tests
    if len(sys.argv) > 1:
        test_cases = sys.argv[1:]
    else:
        # Run all tests in TEST_SUITE by default
        test_cases = list(TEST_SUITE.keys())
        print(f"Running all configured tests: {', '.join(test_cases)}\n", flush=True)
    
    passed, failed = runner.run_all_tests(test_cases)
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()
