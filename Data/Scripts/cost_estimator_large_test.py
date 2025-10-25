# cost_estimator_large_test.py
"""
This script performs a large-scale analysis of VRP solutions for medium and
large instances. It compares the performance of two different VRP solution
heuristics (Clark-Wright and Initial Solution Generator) and evaluates the
solutions' costs using both a precise LKH-3 solver and a machine learning
estimator.

The analysis is parallelized using processes for true parallelism on CPU-bound tasks.
"""
import os
import sys
import time
import math
import joblib
import subprocess
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
import collections
import copy

# --- Ensure required files can be imported ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
# The provided files are in the context of the user's project structure
sys.path.append(str(SCRIPT_DIR))

try:
    from vrp_utils import (
        parse_vrp,
        calculate_estimated_cost,
        get_true_VRP_cost,
    )
    from initial_solution_generator import generate_feasible_solution_regret
    from heuristic_engine import PartitionState # Needed for initial solution costing
except ImportError as e:
    print(f"FATAL: Could not import a required module. Error: {e}")
    sys.exit(1)

# --- Configuration ---
ML_MODEL_DIR = SCRIPT_DIR / "ML_model"
ALPHA_BETA_MODEL_PATH = ML_MODEL_DIR / "alpha_beta_predictor_model.joblib"
OUTPUT_DIR = DATA_DIR / "solution_analysis"
INSTANCES_DIR_MEDIUM = DATA_DIR / "instances_medium"
INSTANCES_DIR_LARGE = DATA_DIR / "instances_large"

# --- Clark-Wright Savings Heuristic Implementation ---

def solve_clark_wright(customer_data, depot_id, depot_coord, capacity):
    """
    Implements the Clark-Wright savings algorithm.
    It's a greedy approach that merges routes with the highest cost savings.
    """
    customer_ids = list(customer_data.keys())
    routes = {cid: [cid] for cid in customer_ids}
    route_demands = {cid: customer_data[cid]['demand'] for cid in customer_ids}
    
    savings = []
    coords = {**{depot_id: depot_coord}, **{cid: data['coords'] for cid, data in customer_data.items()}}
    
    for i, j in combinations(customer_ids, 2):
        s_ij = np.linalg.norm(np.array(coords[depot_id]) - np.array(coords[i])) + \
               np.linalg.norm(np.array(coords[depot_id]) - np.array(coords[j])) - \
               np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
        savings.append((s_ij, i, j))
    
    savings.sort(key=lambda x: x[0], reverse=True)
    
    # Track which customer is an endpoint for a route for O(1) lookups
    endpoints = {cid: (cid, cid) for cid in customer_ids} # (start_node, end_node)
    
    for _, i, j in savings:
        route_i_id = next((k for k, v in routes.items() if i in v), None)
        route_j_id = next((k for k, v in routes.items() if j in v), None)
        
        if route_i_id and route_j_id and route_i_id != route_j_id:
            # Check for common endpoints
            i_is_start_of_i = (endpoints[route_i_id][0] == i)
            i_is_end_of_i = (endpoints[route_i_id][1] == i)
            j_is_start_of_j = (endpoints[route_j_id][0] == j)
            j_is_end_of_j = (endpoints[route_j_id][1] == j)
            
            # This is a key part of Clark-Wright: only merge if i and j are endpoints
            if not (i_is_start_of_i or i_is_end_of_i): continue
            if not (j_is_start_of_j or j_is_end_of_j): continue
            
            new_demand = route_demands[route_i_id] + route_demands[route_j_id]
            if new_demand <= capacity:
                route_i = routes[route_i_id]
                route_j = routes[route_j_id]
                new_route = None
                
                # Check all four possible endpoint connections
                if i_is_end_of_i and j_is_start_of_j:
                    new_route = route_i + route_j
                elif i_is_start_of_i and j_is_end_of_j:
                    new_route = route_j + route_i
                elif i_is_end_of_i and j_is_end_of_j:
                    new_route = route_i + route_j[::-1]
                elif i_is_start_of_i and j_is_start_of_j:
                    new_route = route_i[::-1] + route_j

                if new_route:
                    routes[route_i_id] = new_route
                    route_demands[route_i_id] = new_demand
                    
                    # Update the new route's endpoints
                    new_start = new_route[0]
                    new_end = new_route[-1]
                    endpoints[route_i_id] = (new_start, new_end)
                    
                    del routes[route_j_id]
                    del endpoints[route_j_id]

    final_partition = {i: route for i, route in enumerate(routes.values())}
    return final_partition

# --- Analysis Function for a Single Instance ---
def analyze_single_instance(instance_path, ml_model):
    """
    Analyzes a single VRP instance by comparing two heuristics and
    two cost methods. Designed to be run in parallel using processes.
    """
    try:
        instance_basename = instance_path.stem
        
        # 1. Parse Instance Data
        capacity, depot_id, coords, demands = parse_vrp(instance_path)
        depot_coord = coords[depot_id]
        customer_data = {
            nid: {'coords': c, 'demand': demands.get(nid, 0)}
            for nid, c in coords.items() if nid != depot_id
        }
        all_customer_data_orig = {
            nid: {'coords': coords[nid], 'demand': demands.get(nid, 0)}
            for nid in coords
        }
        total_demand = sum(demands.values())

        results = []
        
        # --- Heuristic 1: Clark-Wright ---
        cw_start_time = time.perf_counter()
        cw_solution = solve_clark_wright(customer_data, depot_id, depot_coord, capacity)
        cw_time = time.perf_counter() - cw_start_time
        
        # Calculate costs for Clark-Wright solution
        # The ML model must be passed to the cost estimator
        cw_est_cost = calculate_estimated_cost(cw_solution, customer_data, depot_coord, mode='composite_mst', bounding_stats=ml_model)
        cw_true_cost = get_true_VRP_cost(cw_solution, all_customer_data_orig, depot_id)
        
        # Record results
        cw_error = ((cw_est_cost - cw_true_cost) / cw_true_cost * 100) if cw_true_cost > 0 else float('inf')
        results.append({
            'Instance': instance_basename,
            'Solver': 'Clark-Wright',
            'Estimated Cost': cw_est_cost,
            'True Cost (LKH)': cw_true_cost,
            'Est. vs True Error (%)': cw_error,
            'Num Vehicles': len(cw_solution),
            'Time (s)': cw_time,
        })
        
        # --- Heuristic 2: Initial Solution Generator ---
        # Set vehicle count to the theoretical minimum
        num_vehicles = math.ceil(total_demand / capacity)
        isg_start_time = time.perf_counter()
        isg_solution = generate_feasible_solution_regret(customer_data, depot_coord, num_vehicles, capacity)
        isg_time = time.perf_counter() - isg_start_time
        
        # Calculate costs for ISG solution
        isg_est_cost = calculate_estimated_cost(isg_solution, customer_data, depot_coord, mode='composite_mst', bounding_stats=ml_model)
        isg_true_cost = get_true_VRP_cost(isg_solution, all_customer_data_orig, depot_id)
        
        # Record results
        isg_error = ((isg_est_cost - isg_true_cost) / isg_true_cost * 100) if isg_true_cost > 0 else float('inf')
        results.append({
            'Instance': instance_basename,
            'Solver': 'Initial Solution Generator',
            'Estimated Cost': isg_est_cost,
            'True Cost (LKH)': isg_true_cost,
            'Est. vs True Error (%)': isg_error,
            'Num Vehicles': len(isg_solution),
            'Time (s)': isg_time,
        })
        
        return results
    except Exception as e:
        print(f"Error processing {instance_path.name}: {e}")
        return []

# --- Main Execution ---
if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    try:
        ml_model = joblib.load(ALPHA_BETA_MODEL_PATH)
        print("✅ Successfully loaded the ML estimator model.")
    except FileNotFoundError:
        print(f"FATAL: ML model not found at '{ALPHA_BETA_MODEL_PATH}'.")
        sys.exit(1)

    # Gather all instances to process
    all_instances = []
    all_instances.extend(list(INSTANCES_DIR_MEDIUM.glob('XML*.vrp')))
    all_instances.extend(list(INSTANCES_DIR_LARGE.glob('XML*.vrp')))
    
    if not all_instances:
        print("FATAL: No VRP instances found in the specified directories.")
        sys.exit(1)

    print(f"Found {len(all_instances)} instances to analyze.")
    
    results_list = []
    num_workers = os.cpu_count() or 1
    
    # This is the core fix: use ProcessPoolExecutor for true parallelism
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Pass ml_model as a fixed argument to each worker task
        futures = {executor.submit(analyze_single_instance, instance, ml_model): instance for instance in all_instances}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_instances), desc="Analyzing VRP Instances"):
            result = future.result()
            if result:
                results_list.extend(result)
                
    if not results_list:
        print("Analysis completed, but no results were generated.")
        sys.exit(0)

    df = pd.DataFrame(results_list)
    output_path = OUTPUT_DIR / "vrp_solution_analysis.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nAnalysis complete. Results saved to: {output_path}")