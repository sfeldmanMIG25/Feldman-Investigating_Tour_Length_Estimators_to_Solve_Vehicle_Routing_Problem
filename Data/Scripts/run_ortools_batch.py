import os
import sys
import time
import re
import glob
import random
import collections
from pathlib import Path
import concurrent.futures

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# --- Add project directory to path ---
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# --- Import project components ---
# These utilities are needed to parse instances and optimal solutions
from vrp_utils import parse_vrp, parse_sol

# --- Configuration ---
DATA_DIR = SCRIPT_DIR.parent
INSTANCE_DIR = DATA_DIR / "instances"
SOLUTION_DIR = DATA_DIR / "solutions"
BATCH_OUTPUT_DIR = SCRIPT_DIR / "Batch_Run_Results"
CONTROL_CSV_PATH = BATCH_OUTPUT_DIR / "benchmark_instances.csv"
# Set a very large time limit (1 week) to act as "unlimited"
OR_TOOLS_TIMEOUT_SECONDS =60*60  #1 hour same as for my heuristics

def write_ortools_report_file(basename, or_tools_cost, optimal_cost, solve_time, output_dir):
    """
    Writes a solution report in the same format as the custom heuristic
    for compatibility with the aggregator.
    """
    output_path = output_dir / f"{basename}_ortools_solution.txt"
    gap = ((or_tools_cost - optimal_cost) / optimal_cost * 100) if optimal_cost > 0 else 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"===== Solution Report for {basename} (OR-Tools) =====\n")
        f.write(f"Generated on: {time.ctime()}\n")
        
        f.write("\n--- Objective Function Comparison ---\n")
        f.write(f"1. Heuristic Estimated Cost (optimization objective) : {or_tools_cost:,.2f}\n")
        f.write(f"2. Heuristic True Cost (LKH)                       : {or_tools_cost:,.2f}\n")
        f.write(f"3. Optimal VRP Cost (LKH Recalculated)             : {optimal_cost:,.2f}\n")
        f.write(f"4. Optimal Estimated Cost (for gap analysis)       : {optimal_cost:,.2f}\n")

        f.write(f"\nPercentage Difference (True vs Optimal)            : {gap:+.2f}%\n")
        f.write(f"Heuristic Final Gap (Est. vs Optimal Est.)       : {gap:+.2f}%\n")

        f.write("\n\n===== Timing Report =====\n")
        f.write(f"1. Data Loading                   : 0.0000 seconds\n")
        f.write(f"2. Heuristic Search               : {solve_time:,.4f} seconds\n")
        f.write(f"3. Final LKH Calculation          : 0.0000 seconds\n")
        f.write(f"4. Total Time                     : {solve_time:,.4f} seconds\n")

def run_single_ortools_solver(instance_filename):
    """
    Solves a single VRP instance using Google OR-Tools and writes a report file.
    """
    basename = instance_filename.replace('.vrp', '')
    instance_path = INSTANCE_DIR / instance_filename
    solution_path = SOLUTION_DIR / f"{basename}.sol"

    capacity, depot_id, coords, demands = parse_vrp(str(instance_path))
    optimal_routes, optimal_cost = parse_sol(str(solution_path))
    num_vehicles = len(optimal_routes)
    
    scaling_factor = 1000
    
    node_map = sorted(coords.keys())
    locations = [coords[i] for i in node_map]
    dist_matrix = [[0] * len(locations) for _ in range(len(locations))]
    for i in range(len(locations)):
        for j in range(len(locations)):
            dist = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))
            dist_matrix[i][j] = int(dist * scaling_factor)
            
    depot_index = node_map.index(depot_id)
    
    manager = pywrapcp.RoutingIndexManager(len(locations), num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    demand_list = [demands.get(i, 0) for i in node_map]
    def demand_callback(from_index):
        return demand_list[manager.IndexToNode(from_index)]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimension(demand_callback_index, 0, capacity, True, 'Capacity')
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(OR_TOOLS_TIMEOUT_SECONDS)

    start_time = time.time()
    solution = routing.SolveWithParameters(search_parameters)
    solve_time = time.time() - start_time
    
    if solution:
        cost = solution.ObjectiveValue() / scaling_factor
        write_ortools_report_file(basename, cost, optimal_cost, solve_time, BATCH_OUTPUT_DIR)
        return f"Successfully processed {instance_filename}"
    else:
        return f"ERROR: No solution found for {instance_filename}"

def select_benchmark_instances(instance_dir: Path, control_csv_path: Path):
    if control_csv_path.exists():
        print(f"✅ Found control file. Loading instances from '{control_csv_path}'.")
        df = pd.read_csv(control_csv_path)
        return df['instance_filename'].tolist()

    print("Control file not found. Generating new benchmark instance set...")
    random.seed(42)
    grouped_instances = collections.defaultdict(list)
    all_files = list(instance_dir.glob("XML100_*.vrp"))
    
    for instance_path in tqdm(all_files, desc="Parsing Instances for Sampling"):
        match = re.match(r'XML100_(\d)(\d)1(\d)_.*\.vrp', instance_path.name)
        if not match: continue
        try:
            capacity, _, _, _ = parse_vrp(instance_path)
            if capacity < 10: continue
        except Exception:
            continue
        depot_type, customer_type, route_size = match.groups()
        grouped_instances[(depot_type, customer_type, route_size)].append(instance_path.name)
    
    selected_instances = []
    for group_key, instances in sorted(grouped_instances.items()):
        random.shuffle(instances)
        num_to_take = min(2, len(instances))
        selected_instances.extend(instances[:num_to_take])
        
    df = pd.DataFrame({'instance_filename': sorted(selected_instances)})
    control_csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(control_csv_path, index=False)
    
    print(f"\n✅ Generated and saved new benchmark set of {len(selected_instances)} instances to '{control_csv_path}'.")
    return df['instance_filename'].tolist()

def run_aggregator(run_path: Path):
    print("\n" + "="*60 + "\n                  ANALYZING BATCH RUN RESULTS\n" + "="*60)
    report_files = glob.glob(os.path.join(run_path, "*_ortools_solution.txt"))
    if not report_files:
        print(f"No OR-Tools solution reports found in '{run_path}'. Cannot aggregate results.")
        return

    def parse_report_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
        def find_metric(pattern, text):
            match = re.search(pattern, text)
            if match:
                value_str = match.group(1).replace(',', '').strip()
                return float(value_str)
            return np.nan

        patterns = {
            'true_gap': r"Percentage Difference \(True vs Optimal\)\s*:\s*([+-]?[\d,]+\.?\d*)",
            'est_gap': r"Heuristic Final Gap \(Est\. vs Optimal Est\.\)\s*:\s*([+-]?[\d,]+\.?\d*)",
            'true_cost': r"2\. Heuristic True Cost \(LKH\)\s*:\s*([\d,]+\.?\d*)",
            'optimal_cost': r"3\. Optimal VRP Cost \(LKH Recalculated\)\s*:\s*([\d,]+\.?\d*)",
            'total_time': r"4\. Total Time\s*:\s*([\d,]+\.?\d*)",
        }
        data = {key: find_metric(p, content) for key, p in patterns.items()}
        data['filename'] = os.path.basename(filepath)
        if pd.isna(data['true_gap']): return None
        return data

    df = pd.DataFrame([d for d in (parse_report_file(f) for f in report_files) if d is not None])
    if df.empty:
        print("Could not parse any valid data from the solution files.")
        return

    csv_path = os.path.join(run_path, 'summary_results_ortools.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nSuccessfully saved detailed results for {len(df)} instances to '{csv_path}'")

    print("\n" + "="*60 + "\n               OR-TOOLS BATCH PERFORMANCE SUMMARY\n" + "="*60)
    print("\n--- Overall Gap of True Cost from Optimal (%) ---")
    print(df['true_gap'].describe().round(2))
    print("\n--- Overall Total Runtime (seconds) ---")
    print(df['total_time'].describe().round(2))
    print("="*60)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('OR-Tools Performance Distribution (Batch Run)', fontsize=16)
    sns.boxplot(y=df['true_gap'], ax=axes[0], color='#a1c9f4')
    axes[0].set_title('Gap of True Cost from Optimal')
    axes[0].set_ylabel('Gap (%)')
    sns.boxplot(y=df['est_gap'].dropna(), ax=axes[1], color='#b2df8a')
    axes[1].set_title('Final Gap (Estimator)')
    axes[1].set_ylabel('Gap (%)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    box_plot_path = os.path.join(run_path, 'summary_gap_plots_ortools.png')
    plt.savefig(box_plot_path, dpi=600)
    plt.close()
    print(f"\nSummary plots saved in '{box_plot_path}'. Aggregation complete.")


def main():
    BATCH_OUTPUT_DIR.mkdir(exist_ok=True)
    
    instances_to_run = select_benchmark_instances(INSTANCE_DIR, CONTROL_CSV_PATH)
    print(f"\nSelected {len(instances_to_run)} instances for the OR-Tools benchmark run.")

    num_workers = os.cpu_count()
    print(f"Starting parallel batch run on {num_workers} cores. Each instance has a virtually unlimited time limit.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # The list is created to ensure all tasks complete before moving on
        results = list(tqdm(executor.map(run_single_ortools_solver, instances_to_run), total=len(instances_to_run), desc="Running OR-Tools Benchmark"))

    print("\n--- Batch processing complete. ---")

    # Use the results list to provide a summary
    success_count = sum(1 for r in results if r.startswith("Successfully"))
    error_count = len(results) - success_count
    print(f"Summary: {success_count} instances processed successfully, {error_count} failed.")
    
    if error_count > 0:
        print("Errors occurred on the following instances:")
        for r in results:
            if r.startswith("ERROR"):
                print(f"  - {r}")
    
    run_aggregator(BATCH_OUTPUT_DIR)


if __name__ == "__main__":
    main()