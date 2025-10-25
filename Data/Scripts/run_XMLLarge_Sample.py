#!/usr/bin/env python3
# run_XMLLarge_Sample.py
import os
import sys
import time
import re
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import concurrent.futures
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

# --- Part 1: Script Configuration and Setup ---

# Set the mode: 'RUN' to execute the heuristic, 'AGGREGATE' to analyze results
MODE = 'RUN'
SOLVER_CHOICE = 'tabu2'
HEURISTIC_TIMEOUT = 6 * 3600  # 6 hours

# Dynamic Import of Solver Class
if SOLVER_CHOICE == 'tabu2':
    from heuristic_strategy_tabu_2 import VRPInstanceSolverTabu2 as SolverClass
    print(f"✅ Using '{SOLVER_CHOICE}' heuristic: VRPInstanceSolverTabu2")
else:
    raise ValueError(f"This script is configured for SOLVER_CHOICE='tabu2'.")

# Path Definitions
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))
from vrp_utils import parse_vrp

DATA_DIR = SCRIPT_DIR.parent
INSTANCE_DIR_MEDIUM = DATA_DIR / "instances_medium"
INSTANCE_DIR_LARGE = DATA_DIR / "instances_large"
ML_MODEL_PATH = SCRIPT_DIR / "ML_model" / "alpha_predictor_model.joblib"

HEURISTIC_OUTPUT_DIR = SCRIPT_DIR / f"Batch_Run_Results_{SOLVER_CHOICE}_large_GART"
BASE_SOLUTION_DIR = DATA_DIR # Root for solutions_medium_*, solutions_large_*
CONTROL_CSV_PATH = HEURISTIC_OUTPUT_DIR / 'benchmark_instances_xml_large.csv'

# Baseline Heuristics List
BASELINE_SOLVERS = ['ClarkWright', 'KMeans', 'NearestNeighbor', 'OrTools', 'Vroom']


# --- Part 2: Instance Sampling and Control File Generation ---

def create_benchmark_control_file():
    """
    Scans instance directories, creates a representative sample (1 of each type),
    and saves it to a control file for reproducible runs.
    """
    print("--- Generating new benchmark control file ---")
    sampled_instances = []
    instance_dirs_to_scan = [INSTANCE_DIR_MEDIUM, INSTANCE_DIR_LARGE]
    
    config_pattern = re.compile(r"XML(\d+)_(\d)(\d)(\d)_([\d\.]+)_(\d+)\.vrp")

    for inst_dir in instance_dirs_to_scan:
        if not inst_dir.exists():
            print(f"⚠️ Warning: Instance directory not found, skipping: {inst_dir}")
            continue

        instance_types = {}
        vrp_files = sorted(list(inst_dir.glob('*.vrp')))

        for f_path in vrp_files:
            match = config_pattern.match(f_path.name)
            if not match:
                continue
            
            # (n, d_pos, c_pos, demand_type, capacity_factor)
            config_tuple = (
                match.group(1), match.group(2), match.group(3), 
                match.group(4), match.group(5)
            )
            
            if config_tuple not in instance_types:
                instance_types[config_tuple] = f_path.name

        sampled_instances.extend(instance_types.values())

    if not sampled_instances:
        print("❌ FATAL: No instances found to create a control file. Please generate instances first.")
        sys.exit(1)

    df = pd.DataFrame(sampled_instances, columns=['instance_filename'])
    HEURISTIC_OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(CONTROL_CSV_PATH, index=False)
    print(f"✅ Saved {len(df)} sampled instances to '{CONTROL_CSV_PATH}'")


# --- Part 3: Best-Known Solution Discovery ---

def find_best_known_solution(instance_basename: str):
    """
    Searches all baseline solution directories for a given instance and
    returns the path and cost of the best one found.
    """
    best_cost = float('inf')
    best_sol_path = None
    
    search_dirs = glob.glob(str(BASE_SOLUTION_DIR / "solutions_*_*"))
    
    for sol_dir in search_dirs:
        sol_path = Path(sol_dir) / f"{instance_basename}.sol"
        if sol_path.exists():
            try:
                with open(sol_path, 'r') as f:
                    for line in f:
                        if line.lower().strip().startswith("cost:"):
                            current_cost = float(line.split(":")[1].strip())
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_sol_path = str(sol_path)
                            break
            except (ValueError, IndexError):
                print(f"⚠️ Warning: Could not parse cost from {sol_path}")

    return best_sol_path, best_cost


# --- Part 4: Heuristic Execution (RUN mode) ---

def run_single_solver(instance_filename: str, solver_class, common_params: dict):
    """Worker function to solve a single VRP instance in a separate process."""
    basename = instance_filename.replace('.vrp', '')
    print(f"  [Worker] Starting to process: {instance_filename}")

    best_sol_path, _ = find_best_known_solution(basename)
    if not best_sol_path:
        return f"ERROR: No best-known solution file found for {basename}. Cannot run heuristic."
    
    # Determine correct instance directory
    if "10000" in instance_filename:
        instance_dir = INSTANCE_DIR_LARGE
    else:
        instance_dir = INSTANCE_DIR_MEDIUM

    solver_params = {
        **common_params,
        'instance_path': str(instance_dir / instance_filename),
        'solution_path': best_sol_path
    }
    solver = solver_class(**solver_params)
    solver.solve()
    return f"Successfully processed {instance_filename}"


# --- Part 5: Results Aggregation and Comparison (AGGREGATE mode) ---

def parse_baseline_solution(sol_path: Path, best_known_cost: float):
    """Parses a simple .sol file from the initial generator."""
    instance = sol_path.stem
    solver = sol_path.parent.name.replace('solutions_medium_', '').replace('solutions_large_', '')
    parsed_cost, time_taken = np.nan, np.nan

    with open(sol_path, 'r') as f:
        for line in f:
            if line.lower().startswith("cost:"):
                parsed_cost = float(line.split(":")[1].strip())
            elif line.lower().startswith("time:"):
                time_taken = float(line.split(":")[1].strip())
    
    true_gap = ((parsed_cost - best_known_cost) / best_known_cost * 100) if best_known_cost > 0 else 0.0

    return {
        'instance': instance,
        'solver': solver.capitalize(),
        'Heuristic_True_Cost': parsed_cost,
        'Heuristic_Estimated_Cost': np.nan, # Not applicable for baselines
        'Optimal_Cost': best_known_cost,
        'True_Gap': true_gap,
        'Estimator_Gap': np.nan, 
        'Total_Time': time_taken
    }

def parse_tabu2_report(report_path: Path):
    """Parses the detailed report from VRPInstanceSolverTabu2."""
    instance = report_path.name.replace('_solution.txt', '')
    with open(report_path, 'r') as f:
        content = f.read()

    patterns = {
        'Heuristic_Estimated_Cost': r"Heuristic Estimated Cost \(optimization objective\)\s*:\s*([\d,]+\.\d+)",
        'Heuristic_True_Cost': r"Heuristic True Cost \(LKH\)\s*:\s*([\d,]+\.\d+)",
        'Optimal_Cost': r"Optimal VRP Cost \(LKH Recalculated\)\s*:\s*([\d,]+\.\d+)",
        'True_Gap': r"Percentage Difference \(True vs Optimal\)\s*:\s*([+-]?\d+\.\d+)",
        'Estimator_Gap': r"Heuristic Final Gap \(Est\. vs Optimal Est\.\)\s*:\s*([+-]?\d+\.\d+)",
        'Total_Time': r"Total Time\s*:\s*([\d,]+\.\d+)"
    }
    
    result = {'instance': instance, 'solver': SOLVER_CHOICE}
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        result[key] = float(match.group(1).replace(',', '')) if match else np.nan
    
    return result

def aggregate_and_compare_results():
    """Aggregates results from all baseline solvers and tabu2, then reports and plots."""
    print("\n" + "="*60)
    print("--- Starting Comparative Aggregation ---")
    print("="*60)

    if not CONTROL_CSV_PATH.exists():
        print(f"❌ Control file not found at '{CONTROL_CSV_PATH}'. Cannot aggregate.")
        return

    instances_df = pd.read_csv(CONTROL_CSV_PATH)
    all_results = []

    # Loop 1: Process Baseline Heuristics
    print(" > Parsing baseline heuristic solutions...")
    for instance_file in tqdm(instances_df['instance_filename'], desc="Parsing Baselines"):
        basename = instance_file.replace('.vrp', '')
        _, best_known_cost = find_best_known_solution(basename)
        if best_known_cost == float('inf'):
            continue

        size_folder = "large" if "10000" in basename else "medium"
        for solver_name in BASELINE_SOLVERS:
            sol_path = BASE_SOLUTION_DIR / f"solutions_{size_folder}_{solver_name.lower()}" / f"{basename}.sol"
            if sol_path.exists():
                parsed_data = parse_baseline_solution(sol_path, best_known_cost)
                all_results.append(parsed_data)

    # Loop 2: Process tabu2 Heuristic
    print(f" > Parsing '{SOLVER_CHOICE}' heuristic reports...")
    for instance_file in tqdm(instances_df['instance_filename'], desc=f"Parsing {SOLVER_CHOICE}"):
        basename = instance_file.replace('.vrp', '')
        report_path = HEURISTIC_OUTPUT_DIR / f"{basename}_solution.txt"
        if report_path.exists():
            parsed_data = parse_tabu2_report(report_path)
            all_results.append(parsed_data)

    if not all_results:
        print("❌ No result files found to aggregate.")
        return

    # Data Consolidation and Reporting
    df_results = pd.DataFrame(all_results)
    excel_path = HEURISTIC_OUTPUT_DIR / "summary_results_large_comparison.xlsx"
    df_results.to_excel(excel_path, index=False, float_format="%.2f")
    print(f"\n✅ Full comparative results saved to '{excel_path}'")

    # Terminal Statistics
    print("\n" + "="*60)
    print("--- Performance Statistics (Absolute Costs) ---")
    print("="*60)
    print(df_results.groupby('solver')[['Heuristic_True_Cost', 'Heuristic_Estimated_Cost']].describe().round(2))

    print("\n" + "="*60)
    print("--- Performance Statistics (Gap %) ---")
    print("="*60)
    print(df_results.groupby('solver')[['True_Gap', 'Estimator_Gap']].describe().round(2))
    
    # Plot Generation
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Box Plot (True Gap)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_results, x='solver', y='True_Gap', hue='solver', palette='viridis', legend=False)
    plt.title('Solution Quality Comparison (True Gap from Best-Known)', fontsize=16)
    plt.ylabel('Percentage Gap from Best-Known (%)')
    plt.xlabel('Solver')
    plt.xticks(rotation=45, ha='right')
    gap_plot_path = HEURISTIC_OUTPUT_DIR / "summary_gap_boxplot_large.png"
    plt.tight_layout(); plt.savefig(gap_plot_path, dpi=600); plt.close()
    print(f"\n✅ Comparative gap box plot saved to '{gap_plot_path}'")

    # Plot 2: Box Plot (Time)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_results, x='solver', y='Total_Time', hue='solver', palette='plasma', legend=False)
    plt.yscale('log')
    plt.title('Computational Time Comparison', fontsize=16)
    plt.ylabel('Total Time (seconds, log scale)')
    plt.xlabel('Solver')
    plt.xticks(rotation=45, ha='right')
    time_plot_path = HEURISTIC_OUTPUT_DIR / "summary_time_boxplot_large.png"
    plt.tight_layout(); plt.savefig(time_plot_path, dpi=600); plt.close()
    print(f"✅ Comparative time box plot saved to '{time_plot_path}'")

    # Plot 3: Scatter Plot (tabu2 specific)
    df_tabu = df_results[df_results['solver'] == SOLVER_CHOICE].dropna(subset=['Heuristic_Estimated_Cost', 'Heuristic_True_Cost'])
    if not df_tabu.empty:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_tabu, x='Heuristic_Estimated_Cost', y='Heuristic_True_Cost', s=100, color='crimson')
        plt.axline((0, 0), slope=1, color='gray', linestyle='--', label='y=x (Perfect Estimator)')
        plt.title(f'Estimated Cost vs. True Cost ({SOLVER_CHOICE})', fontsize=16)
        plt.xlabel('Final Estimated Cost')
        plt.ylabel('Final True Cost (LKH)')
        plt.grid(True, which='both', linestyle='--')
        plt.legend()
        scatter_plot_path = HEURISTIC_OUTPUT_DIR / "summary_scatter_plot_tabu2_large.png"
        plt.tight_layout(); plt.savefig(scatter_plot_path, dpi=600); plt.close()
        print(f"✅ Estimated vs. True Cost scatter plot saved to '{scatter_plot_path}'")


# --- Part 6: Main Execution Block ---

def main():
    """Orchestrates the script's execution based on the MODE constant."""
    if MODE == 'RUN':
        HEURISTIC_OUTPUT_DIR.mkdir(exist_ok=True)
        if not CONTROL_CSV_PATH.exists():
            create_benchmark_control_file()
        
        instances_to_run = pd.read_csv(CONTROL_CSV_PATH)['instance_filename'].tolist()
        ml_model = joblib.load(ML_MODEL_PATH)
        common_params = {
            'output_dir': str(HEURISTIC_OUTPUT_DIR),
            'heuristic_timeout': HEURISTIC_TIMEOUT,
            'ml_model': ml_model,
            'apply_offset': False # FIX: Do not apply +1 offset to customer IDs
        }
        num_workers = os.cpu_count()
        
        print(f"\nStarting parallel batch run for {len(instances_to_run)} instances on {num_workers} cores.")
        
        task_function = partial(run_single_solver, solver_class=SolverClass, common_params=common_params)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(task_function, instances_to_run), total=len(instances_to_run), desc="Running Benchmark"))
        
        for res in results:
            if "ERROR" in res:
                print(res)

        print("\n--- Batch processing complete. Now aggregating results... ---")
        aggregate_and_compare_results()

    elif MODE == 'AGGREGATE':
        aggregate_and_compare_results()

    else:
        print(f"Invalid MODE: '{MODE}'. Choose 'RUN' or 'AGGREGATE'.")

if __name__ == "__main__":
    main()