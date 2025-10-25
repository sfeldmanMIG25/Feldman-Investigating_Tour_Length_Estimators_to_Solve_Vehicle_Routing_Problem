#!/usr/bin/env python3
# run_tabu_proof_of_concept.py

import os
import sys
import time
import re
import glob
import copy
import math
import collections
import traceback
import gc
from pathlib import Path
from itertools import combinations
from functools import partial, lru_cache

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt

# --- Ensure vrp_utils is importable ---
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

from vrp_utils import (
    parse_vrp, parse_sol, get_true_VRP_cost,
    write_solution_file, plot_solution, verify_solution_feasibility,
    estimate_tsp_tour_length
)

class VRPInstanceSolverTabuSimple:
    """
    Implements a simple Tabu Search heuristic for the VRP.
    
    The search operates on partitions of customers, not ordered routes. Its goal
    is to find a local optimum for a given TSP cost estimator, starting from a
    high-quality initial solution (e.g., the known VRP optimum).
    
    The neighborhood consists of both relocate (1-0) and swap (1-1) moves.
    """
    def __init__(self, instance_path: str, solution_path: str, output_dir: str, ml_model: any,
                 heuristic_timeout: int = 7200, apply_offset: bool = True, **kwargs):
        
        self.instance_path = instance_path
        self.solution_path = solution_path
        self.output_dir = output_dir
        self.basename = os.path.splitext(os.path.basename(instance_path))[0]
        self.ml_model = ml_model
        self.heuristic_timeout = heuristic_timeout
        self.apply_offset = apply_offset

        # --- Search Parameters & State ---
        self.tabu_list = None
        self.tabu_tenure = 0
        
        # --- Problem Data ---
        self.capacity = 0
        self.depot_id = 0
        self.depot_coord = (0, 0)
        self.customers = {}
        self.all_customer_data_orig = {}
        
        # --- Solution Tracking ---
        self.current_solution = {}
        self.current_cost = float('inf')
        self.global_best_solution = None
        self.global_best_cost = float('inf')
        
        # --- Optimal Solution Data for Reporting ---
        self.optimal_routes = []
        self.optimal_cost = 0
        self.optimal_estimated_cost = 0

        # --- Reporting & Analysis ---
        self.timing_results = {}
        self.estimator_gap_history = []

    @lru_cache(maxsize=8192)
    def _get_cached_route_cost(self, node_ids_tuple: tuple) -> float:
        if not node_ids_tuple: return 0.0
        nodes_in_route_coords = [self.customers[nid]['coords'] for nid in node_ids_tuple] + [self.depot_coord]
        return estimate_tsp_tour_length(nodes_in_route_coords, mode='RegressionTree', bounding_stats=self.ml_model)

    def solve(self):
        """Main orchestration method to run the entire solution process."""
        overall_start_time = time.time()
        
        self._load_and_initialize_data()
        if not self.current_solution:
            print(f"FATAL: Could not initialize a starting solution for {self.basename}. Aborting.")
            return

        self.timing_results['2. Tabu Search'] = time.time()
        self._run_tabu_search()
        self.timing_results['2. Tabu Search'] = time.time() - self.timing_results['2. Tabu Search']

        self.timing_results['3. Final LKH Calculation'] = time.time()
        best_true_cost = get_true_VRP_cost(self.global_best_solution, self.all_customer_data_orig, self.depot_id)
        self.timing_results['3. Final LKH Calculation'] = time.time() - self.timing_results['3. Final LKH Calculation']

        self._generate_reports(overall_start_time, best_true_cost)
        gc.collect()

    def _load_and_initialize_data(self):
        step_start_time = time.time()
        self.capacity, self.depot_id, coords, demands = parse_vrp(self.instance_path)
        self.depot_coord = coords[self.depot_id]
        self.all_customer_data_orig = {nid: {'coords': c, 'demand': demands.get(nid, 0)} for nid, c in coords.items()}
        self.customers = {nid: v for nid, v in self.all_customer_data_orig.items() if nid != self.depot_id}

        self.optimal_routes, self.optimal_cost = parse_sol(self.solution_path, apply_offset=self.apply_offset)
        optimal_partition = {i: route for i, route in enumerate(self.optimal_routes)}
        self.optimal_estimated_cost = self._calculate_total_estimated_cost(optimal_partition)
        
        # --- Proof of Concept: Initialize with the optimal solution ---
        self.current_solution = copy.deepcopy(optimal_partition)
        self.global_best_solution = copy.deepcopy(optimal_partition)
        self.current_cost = self.optimal_estimated_cost
        self.global_best_cost = self.optimal_estimated_cost
        
        # --- Initialize Tabu Search Parameters ---
        num_customers = len(self.customers)
        self.tabu_tenure = int(math.sqrt(num_customers)) if num_customers > 0 else 10
        self.tabu_list = collections.deque(maxlen=self.tabu_tenure)

        self._log_gap_event(0.0, self.current_cost, 'initial_optimal')
        print(f"[{self.basename}] Initialized with optimal solution. Est. Cost: {self.current_cost:,.2f}, Tabu Tenure: {self.tabu_tenure}")
        self.timing_results['1. Data Loading'] = time.time() - step_start_time

    def _run_tabu_search(self):
        start_time = time.time()
        iteration_counter = 0

        while (time.time() - start_time) < self.heuristic_timeout:
            iteration_counter += 1
            best_move = None
            best_move_delta = float('inf')
            
            # --- 1. Find Best Relocation Move ---
            best_relocation = self._find_best_relocation_move()
            
            # --- 2. Find Best Swap Move ---
            best_swap = self._find_best_swap_move()

            # --- 3. Select Best Overall Move ---
            if best_relocation and best_relocation['delta'] < best_move_delta:
                best_move_delta = best_relocation['delta']
                best_move = best_relocation

            if best_swap and best_swap['delta'] < best_move_delta:
                best_move_delta = best_swap['delta']
                best_move = best_swap
            
            # --- 4. Execute Move or Terminate ---
            if best_move is None:
                print(f"  [{self.basename}] Search stagnated. No valid moves found. Terminating.")
                break

            self._execute_move(best_move)

            # Periodically recalculate the cost from scratch to prevent float error drift
            if iteration_counter % 1000 == 0:
                self.current_cost = self._calculate_total_estimated_cost(self.current_solution)

            self._log_gap_event(time.time() - start_time, self.current_cost, 'tabu_move')

            if self.current_cost < self.global_best_cost:
                self.global_best_cost = self.current_cost
                self.global_best_solution = copy.deepcopy(self.current_solution)
                gap_percent = self.estimator_gap_history[-1][1] if self.estimator_gap_history else 0.0
                print(f"  > New Best @ {time.time() - start_time:.2f}s | Est. Cost: {self.global_best_cost:,.2f} (Gap: {gap_percent:+.2f}%)")

    def _find_best_relocation_move(self):
        best_relocation = None
        min_delta = float('inf')
        
        route_demands = {vid: sum(self.customers[n]['demand'] for n in route) for vid, route in self.current_solution.items()}

        for from_vid, route in self.current_solution.items():
            for node in route:
                node_demand = self.customers[node]['demand']
                for to_vid in self.current_solution.keys():
                    if from_vid == to_vid: continue
                    
                    if route_demands[to_vid] + node_demand <= self.capacity:
                        delta = self._calculate_relocate_delta(node, from_vid, to_vid, self.current_solution)
                        
                        tabu_attr = (node, from_vid)
                        is_tabu = tabu_attr in self.tabu_list
                        
                        if is_tabu and (self.current_cost + delta) >= self.global_best_cost:
                            continue

                        if delta < min_delta:
                            min_delta = delta
                            best_relocation = {
                                'type': 'relocate',
                                'node': node,
                                'from': from_vid,
                                'to': to_vid,
                                'delta': delta,
                                'tabu_attr': (node, to_vid)
                            }
        return best_relocation

    def _find_best_swap_move(self):
        best_swap = None
        min_delta = float('inf')
        route_demands = {vid: sum(self.customers[n]['demand'] for n in route) for vid, route in self.current_solution.items()}

        for v1_id, v2_id in combinations(self.current_solution.keys(), 2):
            for n1 in self.current_solution[v1_id]:
                for n2 in self.current_solution[v2_id]:
                    d1, d2 = self.customers[n1]['demand'], self.customers[n2]['demand']
                    
                    if (route_demands[v1_id] - d1 + d2 <= self.capacity and
                        route_demands[v2_id] - d2 + d1 <= self.capacity):
                        
                        delta = self._calculate_swap_delta(n1, n2, v1_id, v2_id, self.current_solution)
                        
                        tabu_attr = frozenset({n1, n2})
                        is_tabu = tabu_attr in self.tabu_list

                        if is_tabu and (self.current_cost + delta) >= self.global_best_cost:
                            continue
                            
                        if delta < min_delta:
                            min_delta = delta
                            best_swap = {
                                'type': 'swap',
                                'nodes': (n1, n2),
                                'vids': (v1_id, v2_id),
                                'delta': delta,
                                'tabu_attr': tabu_attr
                            }
        return best_swap

    def _execute_move(self, move):
        self.current_cost += move['delta']
        
        if move['type'] == 'relocate':
            node, from_vid, to_vid = move['node'], move['from'], move['to']
            self.current_solution[from_vid].remove(node)
            self.current_solution[to_vid].append(node)
        elif move['type'] == 'swap':
            (n1, n2), (v1, v2) = move['nodes'], move['vids']
            self.current_solution[v1].remove(n1)
            self.current_solution[v1].append(n2)
            self.current_solution[v2].remove(n2)
            self.current_solution[v2].append(n1)
            
        self.tabu_list.append(move['tabu_attr'])

    def _calculate_total_estimated_cost(self, solution):
        return sum(self._get_cached_route_cost(tuple(sorted(nodes))) for nodes in solution.values() if nodes)

    def _calculate_relocate_delta(self, node, from_vid, to_vid, solution):
        c_from, c_to = tuple(sorted(solution[from_vid])), tuple(sorted(solution[to_vid]))
        cost_before = self._get_cached_route_cost(c_from) + self._get_cached_route_cost(c_to)
        c_from_after = tuple(sorted([n for n in c_from if n != node]))
        c_to_after = tuple(sorted(list(c_to) + [node]))
        cost_after = self._get_cached_route_cost(c_from_after) + self._get_cached_route_cost(c_to_after)
        return cost_after - cost_before

    def _calculate_swap_delta(self, n1, n2, v1, v2, solution):
        c1, c2 = tuple(sorted(solution[v1])), tuple(sorted(solution[v2]))
        cost_before = self._get_cached_route_cost(c1) + self._get_cached_route_cost(c2)
        c1_after = tuple(sorted([n for n in c1 if n != n1] + [n2]))
        c2_after = tuple(sorted([n for n in c2 if n != n2] + [n1]))
        cost_after = self._get_cached_route_cost(c1_after) + self._get_cached_route_cost(c2_after)
        return cost_after - cost_before

    def _log_gap_event(self, elapsed_time, current_cost, event_type):
        if self.optimal_estimated_cost == 0:
            gap = 0.0
        else:
            gap = ((current_cost - self.optimal_estimated_cost) / self.optimal_estimated_cost * 100)
        self.estimator_gap_history.append((elapsed_time, gap, event_type))

    def _generate_reports(self, overall_start_time: float, final_true_cost: float):
        final_gap = self.estimator_gap_history[-1][1] if self.estimator_gap_history else 0.0
        self.timing_results['4. Total Time'] = time.time() - overall_start_time
        
        output_path_txt = os.path.join(self.output_dir, f"{self.basename}_solution.txt")
        write_solution_file(
            self.basename, self.global_best_solution, self.global_best_cost, final_true_cost,
            self.optimal_routes, self.optimal_cost, output_path_txt, self.timing_results,
            "TabuSimple", opt_est_cost=self.optimal_estimated_cost, final_gap=final_gap
        )
        output_path_map = os.path.join(self.output_dir, f"{self.basename}_map.png")
        plot_solution(
            self.basename, self.depot_coord, self.global_best_solution, self.customers,
            output_path_map, self.optimal_cost, self.global_best_cost, final_true_cost,
            "TabuSimple", optimal_routes=self.optimal_routes
        )
        output_path_gap = os.path.join(self.output_dir, f"{self.basename}_gap_trend.png")
        self._plot_gap_trend(self.basename, self.estimator_gap_history, output_path_gap, "TabuSimple")
        
        verify_solution_feasibility(self.global_best_solution, self.customers, self.capacity, self.basename)

    def _plot_gap_trend(self, basename, gap_history, output_path, algo_name):
        if not gap_history: return
        df = pd.DataFrame(gap_history, columns=['time', 'gap', 'type'])
        plt.figure(figsize=(12, 7))
        
        event_styles = {
            'initial_optimal': {'color': 'limegreen', 'marker': '*', 's': 300, 'label': 'Initial (Optimal)', 'zorder': 10},
            'tabu_move': {'color': 'blue', 'marker': '.', 's': 40, 'label': 'Tabu Move', 'zorder': 5}
        }
        
        plt.plot(df['time'], df['gap'], color='silver', linestyle='-', linewidth=1, zorder=3)
        for event_type, style in event_styles.items():
            subset = df[df['type'] == event_type]
            if not subset.empty: plt.scatter(subset['time'], subset['gap'], **style)
        
        new_bests = df[df['gap'] < df['gap'].shift(1).fillna(float('inf'))]
        new_bests = new_bests[new_bests['type'] != 'initial_optimal']
        plt.scatter(new_bests['time'], new_bests['gap'], color='red', marker='^', s=150, label='New Best (Estimator)', zorder=10)
        
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Optimal Estimated Cost')
        plt.title(f'Estimator Convergence Profile for {basename}\n({algo_name})', fontsize=15)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Percentage Gap (%)', fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


# --- Runner Script Logic ---

# --- Part 1: Configuration ---
HEURISTIC_TIMEOUT = 2 * 3600  # 2 hours
SOLVER_CHOICE = 'tabu_simple'
DATA_DIR = SCRIPT_DIR.parent
ML_MODEL_PATH = SCRIPT_DIR / "ML_model" / "alpha_predictor_model.joblib"
HEURISTIC_OUTPUT_DIR = SCRIPT_DIR / f"Batch_Run_Results_{SOLVER_CHOICE}"

# Config for XML100 dataset
XML100_INSTANCE_DIR = DATA_DIR / "instances"
XML100_SOLUTION_DIR = DATA_DIR / "solutions"
XML100_CONTROL_CSV = SCRIPT_DIR / "Batch_Run_Results" / "benchmark_instances.csv"


# --- Part 2: Helper Functions for Runner ---

def run_single_solver(task_params: dict, solver_class, common_params: dict):
    """Worker function to solve one VRP instance."""
    instance_filename = task_params['instance_filename']
    basename = instance_filename.replace('.vrp', '')
    print(f"  [Worker] Starting: {basename}")
    
    try:
        solver_params = {**common_params, **task_params}
        solver = solver_class(**solver_params)
        solver.solve()
        return f"Successfully processed {basename}"
    except Exception:
        return f"FATAL ERROR while processing {basename}:\n{traceback.format_exc()}"

def parse_heuristic_report(report_path: Path):
    """Parses the detailed report from the solver."""
    instance = report_path.name.replace('_solution.txt', '')
    with open(report_path, 'r') as f:
        content = f.read()

    patterns = {
        'Heuristic_Estimated_Cost': r"Heuristic Estimated Cost.*:\s*([\d,]+\.\d+)",
        'Heuristic_True_Cost': r"Heuristic True Cost.*:\s*([\d,]+\.\d+)",
        'Optimal_Cost': r"Optimal VRP Cost.*:\s*([\d,]+\.\d+)",
        'True_Gap': r"Percentage Difference.*:\s*([+-]?\d+\.\d+)",
        'Estimator_Gap': r"Heuristic Final Gap.*:\s*([+-]?\d+\.\d+)",
        'Total_Time': r"Total Time\s*:\s*([\d,]+\.\d+)"
    }
    
    result = {'instance': instance, 'solver': SOLVER_CHOICE}
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        result[key] = float(match.group(1).replace(',', '')) if match else np.nan
    return result

def aggregate_results():
    """Aggregates results from the heuristic run, reports, and plots."""
    print("\n" + "="*50 + f"\n--- Aggregating Results for '{SOLVER_CHOICE}' ---\n" + "="*50)
    
    all_results = []
    report_files = glob.glob(str(HEURISTIC_OUTPUT_DIR / "*_solution.txt"))
    for report_path in tqdm(report_files, desc="Parsing Reports"):
        parsed_data = parse_heuristic_report(Path(report_path))
        all_results.append(parsed_data)

    if not all_results:
        print("No result files found to aggregate.")
        return

    df_results = pd.DataFrame(all_results)
    excel_path = HEURISTIC_OUTPUT_DIR / "summary_results_tabu_simple.xlsx"
    df_results.to_excel(excel_path, index=False, float_format="%.2f")
    print(f"\n Full results saved to '{excel_path}'")

    print("\n--- Performance Statistics (Final Gaps %) ---")
    print(df_results[['True_Gap', 'Estimator_Gap']].describe().round(2))


# --- Part 3: Main Execution Block ---

def main():
    """Orchestrates the entire benchmark run."""
    HEURISTIC_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # --- 1. Collect all tasks from both datasets ---
    tasks = []
    # XML100 Tasks (Small)
    if XML100_CONTROL_CSV.exists():
        df_xml100 = pd.read_csv(XML100_CONTROL_CSV)
        for fname in df_xml100['instance_filename']:
            tasks.append({
                'instance_filename': fname,
                'instance_path': str(XML100_INSTANCE_DIR / fname),
                'solution_path': str(XML100_SOLUTION_DIR / fname.replace('.vrp', '.sol')),
                'apply_offset': True,
                'heuristic_timeout': HEURISTIC_TIMEOUT
            })
        print(f"Loaded {len(df_xml100)} tasks from XML100 dataset.")
    else:
        print(f"XML100 control file not found, skipping: {XML100_CONTROL_CSV}")

    # --- MODIFICATION: Disabled XMLLarge dataset to eliminate XML1000 instances ---
    # if XMLLARGE_CONTROL_CSV.exists():
    #     df_xmllarge = pd.read_csv(XMLLARGE_CONTROL_CSV)
    #     loaded_count = 0
    #     for fname in df_xmllarge['instance_filename']:
    #         basename = fname.replace('.vrp', '')
    #         sol_path = find_best_known_solution(basename)
    #         if not sol_path: continue
    #         
    #         instance_dir = XMLLARGE_INSTANCE_DIR_LARGE if "10000" in fname else XMLLARGE_INSTANCE_DIR_MEDIUM
    #         tasks.append({
    #             'instance_filename': fname,
    #             'instance_path': str(instance_dir / fname),
    #             'solution_path': sol_path,
    #             'apply_offset': False,
    #             'heuristic_timeout': HEURISTIC_TIMEOUT_LARGE
    #         })
    #         loaded_count += 1
    #     print(f"Loaded {loaded_count} tasks from XMLLarge dataset.")
    # else:
    #     print(f"XMLLarge control file not found, skipping: {XMLLARGE_CONTROL_CSV}")
        
    if not tasks:
        print("FATAL: No instances found to run. Aborting.")
        return

    # --- 2. Set up and run in parallel ---
    ml_model = joblib.load(ML_MODEL_PATH)
    common_params = {
        'output_dir': str(HEURISTIC_OUTPUT_DIR),
        'ml_model': ml_model
    }
    
    num_workers = os.cpu_count()
    print(f"\nStarting parallel run for {len(tasks)} instances on {num_workers} cores.")
    
    task_function = partial(run_single_solver, solver_class=VRPInstanceSolverTabuSimple, common_params=common_params)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(task_function, tasks), total=len(tasks), desc="Running Benchmark"))
    
    for res in results:
        if "ERROR" in res:
            print(res)

    # --- 3. Aggregate results ---
    aggregate_results()

if __name__ == "__main__":
    aggregate_results()