"""
VRP Heuristic Utilities (vrp_utils.py)

A collection of standard, reusable functions for parsing VRP data,
calculating costs, generating initial solutions, and reporting results.
This version is enhanced to support gap analysis and standardized reporting
for multiple heuristic types.
"""
import os
import time
import math
from math import inf
from itertools import combinations
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import collections
from functools import lru_cache
import lkh
import sys
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd

# --- Configuration ---
KMEANS_RANDOM_SEED = 42
LKH_EXECUTABLE_PATH = "C:\\LKH\\LKH-3.exe" # This path must be updated to the location of your LKH-3 executable.
VROOM_EXECUTABLE_PATH = "/home/mig25/vroom/bin/vroom"

# --- Data Parsing ---
def parse_vrp(vrp_path):
    """Parses a VRP instance file to extract its properties."""
    capacity, coords, demands, depot_id, mode = None, {}, {}, None, ''
    with open(vrp_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("CAPACITY"): capacity = int(line.split()[-1])
            elif line == "NODE_COORD_SECTION": mode = "coords"
            elif line == "DEMAND_SECTION": mode = "demand"
            elif line == "DEPOT_SECTION": mode = "depot"
            elif mode == "coords" and line.split()[0].isdigit():
                parts = line.split()
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif mode == "demand" and line.split()[0].isdigit():
                parts = line.split()
                demands[int(parts[0])] = int(parts[1])
            elif mode == "depot" and line.split()[0].isdigit():
                depot_id = int(line.split()[0])
    if depot_id is None: raise ValueError(f"DEPOT_SECTION not found in {vrp_path}")
    if capacity is None: raise ValueError(f"CAPACITY not found in {vrp_path}")
    return capacity, depot_id, coords, demands

def parse_sol(sol_path, apply_offset=True):
    """
    Parses a CVRPLIB solution file.
    
    Args:
        sol_path (str or Path): The path to the solution file.
        apply_offset (bool): If True, adds 1 to each customer ID. This is for
                             older CVRPLIB files that use 0-based indexing.
    """
    routes, cost = [], None
    try:
        with open(sol_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(sol_path, 'r', encoding='latin-1') as f:
            content = f.read()    
    
    if apply_offset:
        print(f"Diagnostics: Applying +1 ID offset for file {os.path.basename(sol_path)}.")
    else:
        print(f"Diagnostics: Not applying ID offset for file {os.path.basename(sol_path)}.")

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Route"):
            parts = line.split(":")
            if len(parts) > 1 and parts[1].strip():
                # Apply offset only if the flag is True
                if apply_offset:
                    routes.append([int(x) + 1 for x in parts[1].strip().split() if x.isdigit()])
                else:
                    routes.append([int(x) for x in parts[1].strip().split() if x.isdigit()])
        elif line.startswith("Cost"):
            cost = float(line.split()[-1])
            
    if cost is None: 
        raise ValueError(f"Cost not found in {sol_path}")

    # Sanity check to ensure routes are not empty and contain integers
    if not routes or not all(isinstance(node, int) for route in routes for node in route):
        raise ValueError(f"Invalid route format or empty routes found in {sol_path}")

    return routes, cost


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Cost Estimation ---
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Model 1: Çavdar & Sokol (2015) Distribution-Free Formula ---
def _calculate_cavdar(nodes_coords, a0=2.791, a1=0.2669):
    n = len(nodes_coords)
    if n <= 1:
        return 0.0
    coords = np.asarray(nodes_coords, dtype=float)
    # compute convex-hull area (volume attribute = area in 2D)
    hull = ConvexHull(coords)
    area = hull.volume

    # centroid and coordinate dispersions
    mu     = coords.mean(axis=0)
    stdev  = coords.std(axis=0)

    # mean absolute deviation from centroid along each axis
    abs_dev = np.abs(coords - mu)
    c_bar   = abs_dev.mean(axis=0)

    # std. deviation of those absolute deviations
    cstdev = np.sqrt(np.mean((abs_dev - c_bar)**2, axis=0))

    # main estimation terms
    term1 = a0 * math.sqrt(n * cstdev[0] * cstdev[1])
    term2 = a1 * math.sqrt(n * stdev[0] * stdev[1] * area / (c_bar[0] * c_bar[1]))
    est_large_n = term1 + term2
    # small-n correction
    if n < 1000:
        corr = 0.9325 * math.exp(0.00005298 * n) \
             - 0.2972 * math.exp(-0.01452 * n)
        if corr > 0:
            return est_large_n / corr
    return est_large_n

#---- Model 2: Held_Karp Useful for accurately estimating small instances (n<10) ----
def _calculate_held_karp(coords):
    n = len(coords)
    # 1) Build distance matrix
    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i+1, n):
            d = ((xi - coords[j][0])**2 + (yi - coords[j][1])**2)**0.5
            dist[i][j] = dist[j][i] = d
    # 2) Allocate DP table
    dp = [[inf]*n for _ in range(1<<n)]
    dp[1][0] = 0.0               # starting at city 0
    # 3) Fill masks in increasing popcount
    for r in range(2, n+1):
        for subset in combinations(range(1, n), r-1):
            mask = 1              # always include city 0
            for bit in subset:
                mask |= 1<<bit
            # 4) For each “last” in this subset, compute dp[mask][last]
            for last in subset:
                prev_mask = mask ^ (1<<last)
                best = inf
                rem = prev_mask
                while rem:
                    bit = rem & -rem
                    i   = bit.bit_length() - 1
                    rem ^= bit
                    cand = dp[prev_mask][i] + dist[i][last]
                    if cand < best:
                        best = cand
                dp[mask][last] = best
    # 5) Close the tour
    full = (1<<n) - 1
    ans  = inf
    for last in range(1, n):
        cand = dp[full][last] + dist[last][0]
        if cand < ans:
            ans = cand
    return ans

# --- Model 3: Vinel & Silva (2018) based on Beardwood-Halton-Hammersley --- Tested on instnaces of [10,100]
def _calculate_vinel(nodes_coords, b=0.768):
    n = len(nodes_coords)
    coords = np.asarray(nodes_coords, dtype=float)
    # Compute convex‐hull area (hull.volume == area in 2D)
    hull = ConvexHull(coords)
    area = hull.volume
    # Apply the BHH‐based formula
    return b * math.sqrt(n * area)

# --- Model 4: MST-Based Estimation using Prim's Algorithm ---
def _calculate_mst_length(nodes_coords):
    """
    Calculates the Minimum Spanning Tree length for a set of coordinates.
    This version is OPTIMIZED to use the fast Scipy implementation.
    """
    num_nodes = len(nodes_coords)
    if num_nodes <= 1:
        return 0.0
    
    # 1. Create a distance matrix (this is the O(n^2) step)
    dist_matrix = np.linalg.norm(nodes_coords[:, np.newaxis, :] - nodes_coords[np.newaxis, :, :], axis=2)
    
    # 2. Use Scipy to find the MST from the dense distance matrix
    mst = minimum_spanning_tree(dist_matrix)
    
    # 3. The sum of the sparse matrix's data is the total length of the MST
    return mst.sum()

# --- Estimator Function to get the tour length, determines which models are used and bounds them appropriately ---
def estimate_tsp_tour_length(nodes_coords, mode='composite_mst', bounding_stats=None, **kwargs):
    """
    Estimates the TSP tour length for a set of nodes. 'composite_mst' mode is
    ML-driven and handles both new (alpha-beta) and original (alpha-only) models.
    """
    n = len(nodes_coords)
    if n <= 1:
        return 0.0
    if n == 2:  # Depot + 1 customer
        dist = np.linalg.norm(np.array(nodes_coords[0]) - np.array(nodes_coords[1]))
        return dist * 2
    if n == 3:  # Depot + 2 customers
        # Return the perimeter of the triangle formed by the three points.
        p1 = np.array(nodes_coords[0])
        p2 = np.array(nodes_coords[1])
        p3 = np.array(nodes_coords[2])
        cost = np.linalg.norm(p1 - p2) + np.linalg.norm(p2 - p3) + np.linalg.norm(p3 - p1)
        return cost
    if n <= 10:
        return _calculate_held_karp(nodes_coords)
    if mode == 'composite_mst':
        mst_length = _calculate_mst_length(np.array(nodes_coords)) 
        if n > 100:
            original_estimate =  _calculate_cavdar(np.array(nodes_coords), a0=kwargs.get('a0', 2.791), a1=kwargs.get('a1', 0.2669))
            return max(mst_length, min(2*mst_length, original_estimate))
        
        elif bounding_stats is None:
            raise ValueError("The 'composite_mst' mode requires a loaded ML model passed via 'bounding_stats'.")
        else:
            ml_model = bounding_stats
            features_dict, mst_length = calculate_features_and_mst_length(nodes_coords)

            if mst_length == 0:
                return 0.0

            if isinstance(ml_model, MultiOutputRegressor):
                # Logic for the new Alpha-Beta model
                feature_cols = ml_model.estimators_[0].feature_name_
                feature_df = pd.DataFrame([features_dict])[feature_cols]
                prediction = ml_model.predict(feature_df)
                predicted_alpha, predicted_beta = prediction[0]
                estimated_cost = (predicted_alpha * mst_length) + predicted_beta
            else:
                # Fallback logic for the original single-output Alpha model
                feature_cols = ml_model.feature_name_
                feature_df = pd.DataFrame([features_dict])[feature_cols]
                predicted_alpha = ml_model.predict(feature_df)[0]
                estimated_cost = predicted_alpha * mst_length

            return max(mst_length, estimated_cost)

    elif mode == 'composite':       
        if n < 100:
            original_estimate = 0.0
            original_estimate = _calculate_vinel(np.array(nodes_coords), b=kwargs.get('b', 0.768))
        else:
            original_estimate = 0.0
            original_estimate = _calculate_cavdar(np.array(nodes_coords), a0=kwargs.get('a0', 2.791), a1=kwargs.get('a1', 0.2669))
        
        mst_length = _calculate_mst_length(np.array(nodes_coords))    
        return max(mst_length, min(2*mst_length, original_estimate))
            
    elif mode == "held-karp":
        return _calculate_held_karp(nodes_coords)
    elif mode == 'vinel':
        return _calculate_vinel(np.array(nodes_coords), b=kwargs.get('b', 0.768))
    elif mode == 'cavdar':
        return _calculate_cavdar(np.array(nodes_coords), a0=kwargs.get('a0', 2.791), a1=kwargs.get('a1', 0.2669))
    elif mode == 'RegressionTree':
        if bounding_stats is None:
            raise ValueError("The 'composite_mst' mode requires a loaded ML model passed via 'bounding_stats'.")
        else:
            ml_model = bounding_stats
            features_dict, mst_length = calculate_features_and_mst_length(nodes_coords)

            if mst_length == 0:
                return 0.0

            if isinstance(ml_model, MultiOutputRegressor):
                # Logic for the new Alpha-Beta model
                feature_cols = ml_model.estimators_[0].feature_name_
                feature_df = pd.DataFrame([features_dict])[feature_cols]
                prediction = ml_model.predict(feature_df)
                predicted_alpha, predicted_beta = prediction[0]
                estimated_cost = (predicted_alpha * mst_length) + predicted_beta
            else:
                # Fallback logic for the original single-output Alpha model
                feature_cols = ml_model.feature_name_
                feature_df = pd.DataFrame([features_dict])[feature_cols]
                predicted_alpha = ml_model.predict(feature_df)[0]
                estimated_cost = predicted_alpha * mst_length

            return max(mst_length, estimated_cost)
    else:
        raise ValueError(f"Invalid or unsupported mode '{mode}'.")

# This is the updated version of calculate_estimated_cost in vrp_utils.py
def calculate_estimated_cost(partition, customer_data, depot_coord, mode='composite', bounding_stats=None, **kwargs):
    """Calculates the total estimated cost for a solution partition."""
    total_cost = 0.0
    for cluster_nodes in partition.values():
        if not cluster_nodes: 
            continue
        nodes_in_route = [customer_data[nid]['coords'] for nid in cluster_nodes] + [depot_coord]
        # Pass the 'bounding_stats' dictionary down to the core estimator
        total_cost += estimate_tsp_tour_length(nodes_in_route, mode=mode, bounding_stats=bounding_stats, **kwargs)
    return total_cost
#-----------------------------------------------------------------------------------------------------------------------
# --- Complete TSP Calculations
#------------------------------------------------------------------------------------------------------------------------
def solve_tsp_lkh(node_ids_for_tsp, all_customer_data, depot_id):
    """
    Corrected version to call LKH-3 and solve a single tour.
    It now correctly handles node lists and avoids redundant depot entries.
    """
    # Create the node list, ensuring a single, valid depot entry.
    nodes = list(node_ids_for_tsp)
    if depot_id in nodes:
        nodes.remove(depot_id)
    nodes.insert(0, depot_id)  # Correctly ensures the depot is the first node

    num_nodes = len(nodes)
    if num_nodes <= 1:
        return 0.0
    if num_nodes == 2:
        c1_id = nodes[0]
        c2_id = nodes[1]
        c1_coords = all_customer_data[c1_id]['coords']
        c2_coords = all_customer_data[c2_id]['coords']
        dist = np.linalg.norm(np.array(c1_coords) - np.array(c2_coords))
        return dist * 2

    # 1. Create an LKHProblem object
    problem = lkh.LKHProblem()
    problem.name = f'vrp_subproblem_{num_nodes}'
    problem.type = 'TSP'
    problem.dimension = num_nodes
    problem.edge_weight_type = 'EUC_2D'
    
    node_coords = {
        i + 1: all_customer_data[nid]['coords']
        for i, nid in enumerate(nodes)
    }
    
    problem.node_coords = node_coords
    
    try:
        # 2. Solve the problem
        tour_indices = lkh.solve(solver=LKH_EXECUTABLE_PATH, problem=problem, runs=1, max_trials=100)[0]
        
        # 3. Calculate the cost from the returned tour
        tour_cost = 0.0
        for i in range(num_nodes):
            start_node_idx = tour_indices[i]
            end_node_idx = tour_indices[(i + 1) % num_nodes]
            c1 = problem.node_coords[start_node_idx]
            c2 = problem.node_coords[end_node_idx]
            tour_cost += np.linalg.norm(np.array(c1) - np.array(c2))
        
        return tour_cost
    except Exception as e:
        print(f"WARNING: LKH wrapper failed for a subproblem of size {num_nodes}. Error: {e}")
        return 0.0
    
"""Iterate through each created tour to get overall vrp cost"""
def get_true_VRP_cost(partition, all_customer_data, depot_id):
    """Calculates the 'true' cost of a partition by solving the TSP for each cluster."""
    total_recalculated_cost = 0.0
    for _, nodes_unit in sorted(partition.items()):
        if not nodes_unit:
            continue
        # Check for preprocessed node IDs (e.g., '42_0')
        if nodes_unit and isinstance(nodes_unit[0], str):
            nodes_orig = sorted(list(set(int(n.split('_')[0]) for n in nodes_unit)))
        else:
            nodes_orig = sorted(list(set(nodes_unit)))
        
        tour_cost = solve_tsp_lkh(nodes_orig, all_customer_data, depot_id)
        total_recalculated_cost += tour_cost
    
    return total_recalculated_cost


#----------------------------------------------------------------------------------------------------------------
# --- Initialization Support Functions ---
#--------------------------------------------------------------------------------------------------------------------------

def generate_initial_solution_k_means(customer_data, K, Q):
    node_ids, coords_list = list(customer_data.keys()), np.array([data['coords'] for data in customer_data.values()])
    if len(coords_list) < K: K = max(1, len(coords_list))
    if K == 0: return {}
    
    kmeans = KMeans(n_clusters=K, random_state=KMEANS_RANDOM_SEED, n_init=10).fit(coords_list)
    partition = {k: [] for k in range(K)}
    for i, label in enumerate(kmeans.labels_):
        partition[label].append(node_ids[i])
    #Ensure solution has been properly balanced.
    partition = balance_by_node_movement(partition, customer_data, Q)   
    return {k: v for k, v in partition.items() if v}

def generate_single_greedy_route(all_customer_nodes, customer_data, capacity):
    """
    Generates a single, realistic sample route using a greedy nearest-neighbor approach.
    This bypasses full VRP feasibility issues while avoiding pure randomness.
    
    Returns:
        A list of customer IDs representing a single valid route, or None if a seed can't be placed.
    """
    if not all_customer_nodes:
        return None

    unassigned_nodes = set(all_customer_nodes)
    node_coords = {nid: data['coords'] for nid, data in customer_data.items()}

    def get_dist(n1, n2):
        return np.linalg.norm(np.array(node_coords[n1]) - np.array(node_coords[n2]))

    # 1. Pick a random seed customer
    # Ensure the chosen seed can actually fit in the vehicle
    possible_seeds = [n for n in unassigned_nodes if customer_data[n]['demand'] <= capacity]
    if not possible_seeds:
        return None # No single customer can fit in the vehicle
    seed_node = random.choice(possible_seeds)
    
    route = [seed_node]
    current_load = customer_data[seed_node]['demand']
    unassigned_nodes.remove(seed_node)

    # 2. Grow the route by greedily adding the nearest valid neighbor
    while True:
        last_node_in_route = route[-1]
        best_neighbor = None
        min_dist = float('inf')

        # Find the closest neighbor that fits
        for neighbor in unassigned_nodes:
            if current_load + customer_data[neighbor]['demand'] <= capacity:
                dist = get_dist(last_node_in_route, neighbor)
                if dist < min_dist:
                    min_dist = dist
                    best_neighbor = neighbor
        
        if best_neighbor:
            route.append(best_neighbor)
            current_load += customer_data[best_neighbor]['demand']
            unassigned_nodes.remove(best_neighbor)
        else:
            # No more unassigned nodes can fit
            break
            
    return route

def balance_by_node_movement(partition, customer_data, capacity):
    """
    Balances a clustered solution by moving or swapping individual nodes from
    overloaded clusters to clusters with spare capacity.
    Args:
        partition (dict): The current assignment of nodes to clusters.
                          Example: {0: [1, 5, 8], 1: [2, 3, 4]}
        customer_data (dict): A dictionary containing data for each customer node,
                              including 'demand'.
        capacity (int): The maximum capacity for any cluster (vehicle).
    Returns:
        dict: A balanced partition where all cluster loads are within capacity,
              or the best attempt if not fully solvable.
    """
    while True:
        # Calculate current loads and identify overloaded clusters
        cluster_loads = {
            key: sum(customer_data[node]['demand'] for node in nodes)
            for key, nodes in partition.items()
        }
        overloaded_keys = {
            key for key, load in cluster_loads.items() if load > capacity
        }
        # If no clusters are overloaded, the solution is feasible.
        if not overloaded_keys:
            print("Solution is balanced. All capacity constraints met.")
            return {k: v for k, v in partition.items() if v} # Clean up empty clusters
        # --- Main Balancing Loop ---
        made_a_change = False
        # We'll iterate through a copy of the keys since we may modify the original set
        for overloaded_key in list(overloaded_keys):
            # Get all nodes from the overloaded cluster
            nodes_in_cluster = list(partition[overloaded_key])
            # Find the unique demand values present in the cluster and sort them
            # This ensures we always try to move the smallest-demand nodes first.
            demands_to_try = sorted(list({customer_data[n]['demand'] for n in nodes_in_cluster}))
            for demand_val in demands_to_try:
                # Get all nodes with this specific demand and shuffle them to add randomness
                candidate_nodes = [
                    n for n in nodes_in_cluster if customer_data[n]['demand'] == demand_val
                ]
                random.shuffle(candidate_nodes)
                for node_to_move in candidate_nodes:
                    # --- ATTEMPT 1: SWAP (if node demand > 1) ---
                    # Try to swap with a demand-1 node from another vehicle
                    # if it helps the overloaded vehicle and doesn't overload the target.
                    if demand_val > 1:
                        # Find a potential swap partner
                        swap_found = False
                        # Look through all other clusters for a swap target
                        for target_key, target_nodes in partition.items():
                            if target_key == overloaded_key:
                                continue
                            # Check if the target can accept our node after giving up a demand-1 node
                            target_load = cluster_loads[target_key]
                            if (target_load - 1 + demand_val) <= capacity:
                                # Find a node with demand 1 in the target cluster
                                for node_to_swap in target_nodes:
                                    if customer_data[node_to_swap]['demand'] == 1:
                                        # Perform the swap!
                                        partition[overloaded_key].remove(node_to_move)
                                        partition[target_key].append(node_to_move)
                                        
                                        partition[target_key].remove(node_to_swap)
                                        partition[overloaded_key].append(node_to_swap)
                                        
                                        print(f"Swapped Node {node_to_move}(d:{demand_val}) from C:{overloaded_key} with Node {node_to_swap}(d:1) from C:{target_key}")
                                        made_a_change = True
                                        swap_found = True
                                        break # Stop looking for nodes to swap in this cluster
                            if swap_found:
                                break # Stop looking for target clusters
                        if swap_found:
                            break # Move to the next overloaded cluster
                    # --- ATTEMPT 2: MOVE ---
                    # If swap wasn't possible or applicable, try a simple move.
                    move_found = False
                    # Find the first vehicle with enough capacity
                    for target_key, target_nodes in partition.items():
                        if target_key == overloaded_key:
                            continue
                        target_load = cluster_loads[target_key]
                        if target_load + demand_val <= capacity:
                            # Perform the move!
                            partition[overloaded_key].remove(node_to_move)
                            partition[target_key].append(node_to_move)
                            
                            print(f"Moved Node {node_to_move}(d:{demand_val}) from C:{overloaded_key} to C:{target_key}")
                            made_a_change = True
                            move_found = True
                            break # Stop looking for a place to move to
                    if move_found:
                        break # A node was moved, so we restart work on this overloaded cluster
                if made_a_change:
                    break # A change was made, so restart the main 'while' loop
            if made_a_change:
                break # A change was made, so restart the main 'while' loop
        # If we looped through all overloaded clusters and made no changes, we're stuck.
        if not made_a_change:
            print(f"Warning: Could not fully balance the solution. Still {len(overloaded_keys)} overloaded clusters remaining.")
            return {k: v for k, v in partition.items() if v}

def verify_solution_feasibility(solution, customer_data, capacity, instance_name):
    """Checks if a VRP solution is feasible by verifying capacity and customer visit constraints. """
    is_feasible = True
    # 1. Check vehicle capacity constraints
    for vehicle_id, route in solution.items():
        route_demand = sum(customer_data[customer_id]['demand'] for customer_id in route)
        if route_demand > capacity:
            print(f"[{instance_name}] INFEASIBLE: Vehicle {vehicle_id} exceeds capacity. "
                  f"Demand: {route_demand}, Capacity: {capacity}")
            is_feasible = False
    # 2. Check customer assignment constraints (each customer visited exactly once)
    all_assigned_customers = [customer for route in solution.values() for customer in route]
    required_customers = set(customer_data.keys())
    visited_customers = set(all_assigned_customers)
    # Check for missing customers
    missing = required_customers - visited_customers
    if missing:
        print(f"[{instance_name}] INFEASIBLE: The following customers were not assigned to any route: {sorted(list(missing))}")
        is_feasible = False
    # Check for customers not in the original problem (should not happen)
    extra = visited_customers - required_customers
    if extra:
        print(f"[{instance_name}] INFEASIBLE: The following unkown customers were assigned to routes: {sorted(list(extra))}")
        is_feasible = False
    # Check for customers assigned to more than one route
    customer_counts = collections.Counter(all_assigned_customers)
    duplicates = [customer for customer, count in customer_counts.items() if count > 1]
    if duplicates:
        print(f"[{instance_name}] INFEASIBLE: The following customers were assigned to multiple routes: {sorted(duplicates)}")
        is_feasible = False
    if is_feasible:
        print(f"[{instance_name}] Solution is feasible.") 
    return is_feasible

#---------------------------------------------------------------------------------------------------------------------------------------------
# --- Estimator Pre-tuning Functions ---
#---------------------------------------------------------------------------------------------------------------------------------------------

# Replaces 'discover_universal_mst_alpha' in vrp_utils.py
def discover_alpha_samples_from_instance(customer_data, depot_coord, depot_id, capacity, all_customer_data_orig, samples_per_instance=10):
    """
    Generates a set of alpha samples from a single instance to contribute to a universal statistic.
    An alpha sample is the ratio of a route's true cost to its MST length.
    """
    alpha_samples = []
    attempts = 0
    max_attempts = max(1000, samples_per_instance * 50) # Robust attempt limit
    all_customer_nodes = list(customer_data.keys())

    # This function's goal is to return a list of raw alpha values
    while len(alpha_samples) < samples_per_instance and attempts < max_attempts:
        attempts += 1
        tour_nodes = generate_single_greedy_route(all_customer_nodes, customer_data, capacity)
        if not tour_nodes or len(tour_nodes) < 3:
            continue

        nodes_for_tsp = [depot_id] + sorted(list(tour_nodes))
        true_cost = solve_tsp_lkh(nodes_for_tsp, all_customer_data_orig, depot_id)

        coords_for_mst = np.array([depot_coord] + [customer_data[nid]['coords'] for nid in tour_nodes])
        mst_length = _calculate_mst_length(coords_for_mst)

        if true_cost > 0 and mst_length > 0:
            alpha_sample = true_cost / mst_length
            if 1.0 <= alpha_sample <= 2.5: # Widen sanity check slightly
                 alpha_samples.append(alpha_sample)
                 
    return alpha_samples

#--------------------------------------------------------------------------------------------------------------------------------
# --- Post-Processing and Reporting ---
#--------------------------------------------------------------------------------------------------------------------------------------------------
def get_true_heuristic_cost(partition, all_customer_data, depot_id):
    """Calculates the 'true' cost of a partition by solving the TSP for each cluster."""
    total_true_cost = 0.0
    for _, nodes in sorted(partition.items()):
        if not nodes: continue
        node_ids_for_tsp = [depot_id] + sorted(list(set(nodes)))
        if len(node_ids_for_tsp) <= 1: continue
        
        coords_for_tsp = [all_customer_data[nid]['coords'] for nid in node_ids_for_tsp]
        manager = pywrapcp.RoutingIndexManager(len(node_ids_for_tsp), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        dist_matrix = [[int(np.linalg.norm(np.array(c1) - np.array(c2))) for c2 in coords_for_tsp] for c1 in coords_for_tsp]
        def distance_callback(from_index, to_index):
            return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
            
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        solution = routing.SolveWithParameters(search_parameters)
        if solution: total_true_cost += solution.ObjectiveValue()
            
    return total_true_cost

def compare_solutions(heuristic_partition, optimal_routes):
    """Compares heuristic clusters to optimal routes using Jaccard similarity."""
    def jaccard_similarity(set1, set2):
        return len(set1.intersection(set2)) / len(set1.union(set2)) if len(set1.union(set2)) > 0 else 0
        
    report_lines = []
    
    # FIX: Node IDs are now integers, so we just create sets directly.
    heuristic_clusters = {k: set(nodes) for k, nodes in heuristic_partition.items()}
    
    optimal_routes_set = [set(route) for route in optimal_routes]
    
    for k, h_cluster in sorted(heuristic_clusters.items()):
        if not h_cluster: continue
        similarities = [jaccard_similarity(h_cluster, o_route) for o_route in optimal_routes_set]
        best_match_idx, score = (np.argmax(similarities), np.max(similarities)) if similarities else (-1, 0)
        
        report_lines.append(f"\n--- Heuristic Cluster {k+1} (Size: {len(h_cluster)}) ---")
        if best_match_idx != -1:
            best_optimal_route = optimal_routes_set[best_match_idx]
            gained = sorted(list(h_cluster - best_optimal_route))
            lost = sorted(list(best_optimal_route - h_cluster))
            report_lines.append(f"Best matches Optimal Route #{best_match_idx+1} with {score:.2%} similarity.")
            report_lines.append(f"  Nodes Gained ({len(gained)}): {', '.join(map(str, gained)) or 'None'}")
            report_lines.append(f"  Nodes Lost ({len(lost)}): {', '.join(map(str, lost)) or 'None'}")
        else:
            report_lines.append("No matching optimal route found.")
            
    return "\n".join(report_lines)

def write_solution_file(basename, heuristic_partition, est_cost, true_cost, opt_routes, opt_cost, output_path, timing_results, algo_name, **kwargs):
    """Writes a comprehensive solution report file in a standardized format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"===== Solution Report for {basename} ({algo_name}) =====\n")
        f.write(f"Generated on: {time.ctime()}\n")
        
        diff = ((true_cost - opt_cost) / opt_cost * 100) if opt_cost > 0 else 0
        opt_est_cost = kwargs.get('opt_est_cost')
        final_gap = kwargs.get('final_gap')
        
        f.write("\n--- Objective Function Comparison ---\n")
        f.write(f"1. Heuristic Estimated Cost (optimization objective) : {est_cost:,.2f}\n")
        # MODIFICATION 1: Changed text to match aggregator regex 
        f.write(f"2. Heuristic True Cost (LKH)                       : {true_cost:,.2f}\n")
        # MODIFICATION 2: Changed text to match aggregator regex 
        f.write(f"3. Optimal VRP Cost (LKH Recalculated)             : {opt_cost:,.2f}\n")
        if opt_est_cost is not None:
             f.write(f"4. Optimal Estimated Cost (for gap analysis)       : {opt_est_cost:,.2f}\n")

        f.write(f"\nPercentage Difference (True vs Optimal)            : {diff:+.2f}%\n")
        if final_gap is not None:
            f.write(f"Heuristic Final Gap (Est. vs Optimal Est.)       : {final_gap:+.2f}%\n")

        f.write("\n\n===== Heuristic Cluster Composition =====\n")
        # Fix for partition data type (can be unit-demand strings or final integers)
        for k, nodes in sorted(heuristic_partition.items()):
            if not nodes: continue
            orig_nodes = sorted(list(set(nodes)))
            f.write(f"Cluster {k+1} ({len(orig_nodes)} customers):\n  " + ", ".join(map(str, orig_nodes)) + "\n")
            
        f.write("\n\n===== Structural Comparison (vs. Optimal Routes) =====\n")
        f.write(compare_solutions(heuristic_partition, opt_routes))
        
        f.write("\n\n===== Timing Report =====\n")
        for step, duration in timing_results.items():
            f.write(f"{step:<30}: {duration:,.4f} seconds\n")

def plot_solution(basename, depot_coord, partition, customer_data, output_path, opt_cost, est_cost, true_cost, algo_name, optimal_routes=None):
    """
    Generates and saves a plot of the solution clusters.
    MODIFIED: If optimal_routes are provided, it circles nodes that do not
    match their assigned optimal route in black.
    """
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('tab20')

    # --- Start of new logic for circling mismatched nodes ---
    mismatched_coords_to_plot = []
    if optimal_routes:
        def jaccard_similarity(set1, set2):
            return len(set1.intersection(set2)) / len(set1.union(set2)) if len(set1.union(set2)) > 0 else 0
        
        # Create a map of which optimal route each customer belongs to
        optimal_node_to_route_idx_map = {
            node_id: i for i, route in enumerate(optimal_routes) for node_id in route
        }
        
        # Create sets for faster comparison
        heuristic_clusters_orig_sets = {
            k: set(nodes) for k, nodes in partition.items()
        }
        optimal_routes_sets = [set(route) for route in optimal_routes]

        # For each heuristic cluster, find its best-matching optimal route
        for k, h_cluster_nodes in partition.items():
            if not h_cluster_nodes: continue
            
            h_cluster_set = heuristic_clusters_orig_sets.get(k, set())
            if not h_cluster_set: continue

            similarities = [jaccard_similarity(h_cluster_set, o_set) for o_set in optimal_routes_sets]
            matched_opt_route_idx = np.argmax(similarities) if similarities else -1

            if matched_opt_route_idx != -1:
                # Check each node in the cluster
                for original_id in h_cluster_nodes:
                    # If the node's true optimal route doesn't match the cluster's matched route, it's mismatched.
                    if optimal_node_to_route_idx_map.get(original_id) != matched_opt_route_idx:
                        mismatched_coords_to_plot.append(customer_data[original_id]['coords'])
    # --- End of new logic ---

    for k, nodes in partition.items():
        if not nodes: continue
        color = cmap(k % 20)
        coords = np.array([customer_data[nid]['coords'] for nid in nodes])
        plt.scatter(coords[:, 0], coords[:, 1], c=[color], s=30, alpha=0.8, label=f'Cluster {k+1}')

    plt.scatter(depot_coord[0], depot_coord[1], c='red', marker='*', s=250, zorder=10, label='Depot')
    
    # Plot the black circles for mismatched nodes on top
    if mismatched_coords_to_plot:
        mismatched_arr = np.array(mismatched_coords_to_plot)
        plt.scatter(mismatched_arr[:, 0], mismatched_arr[:, 1],
                    facecolors='none', edgecolors='k', linewidths=1.5, s=90, zorder=6,
                    label='Mismatched Node')

    title = (
        f"{algo_name} Solution for {basename}\n"
        f"CVRP Opt Sol: {opt_cost:,.2f} | Heuristic Est: {est_cost:,.2f} | TSP Heuristic Cost: {true_cost:,.2f}"
    )
    plt.title(title, fontsize=14)
    plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), markerscale=2, loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()


def plot_gap_trend(basename, gap_history, output_path, algo_name):
    """Plots the heuristic's gap from the optimal estimate over time."""
    if not gap_history: return
    
    times, gaps = zip(*gap_history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, gaps, marker='.', linestyle='-', markersize=5, color='#9467bd')
    plt.axhline(y=0, color='green', linestyle='--', linewidth=1.5, label='Optimal Estimated Cost')
    
    plt.title(f'Convergence Profile for {basename}\n({algo_name})', fontsize=15)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Gap from Optimal Estimate (%)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def calculate_features_and_mst_length(coords_list):
    """
    Optimized function to calculate the full, expanded feature set and MST length.
    Assumes depot is the first coordinate in the list.
    """
    coords = np.array(coords_list)
    features = {'n': len(coords)}
    
    # Pre-calculate the full distance matrix once
    dist_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    
    # --- Hull Features ---
    try:
        hull = ConvexHull(coords)
        features['convex_hull_area'] = hull.volume
        features['convex_hull_perimeter'] = hull.area
        features['hull_vertex_count'] = len(hull.vertices)
        features['hull_ratio'] = features['hull_vertex_count'] / features['n']
    except Exception:
        features.update({'convex_hull_area': 0, 'convex_hull_perimeter': 0, 'hull_vertex_count': 0, 'hull_ratio': 0})
    
    # --- Bounding Box ---
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    width, height = max_c[0] - min_c[0], max_c[1] - min_c[1]
    features['bounding_box_area'] = width * height
    
    # --- Nearest-Neighbor (NN) Statistics ---
    np.fill_diagonal(dist_matrix, np.inf) # Exclude self-distance
    one_nn_dists = np.min(dist_matrix, axis=1)
    features['one_nn_dist_mean'] = one_nn_dists.mean()
    features['one_nn_dist_std'] = one_nn_dists.std()
    
    # --- PCA for Anisotropy ---
    try:
        pca = PCA(n_components=2).fit(coords)
        eigenvalues = pca.explained_variance_
        features['pca_eigenvalue_ratio'] = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 0
    except Exception:
        features['pca_eigenvalue_ratio'] = 1.0
        
    # --- Expanded MST Features ---
    np.fill_diagonal(dist_matrix, 0) # Restore diagonal for MST
    mst = minimum_spanning_tree(dist_matrix)
    mst_length = mst.sum()
    degrees = np.count_nonzero(mst.toarray() + mst.toarray().T, axis=1)
    features['mst_degree_mean'] = degrees.mean()
    features['mst_degree_max'] = degrees.max()
    features['mst_degree_std'] = degrees.std()
    features['mst_leaf_nodes_fraction'] = np.sum(degrees == 1) / features['n']

    # --- Other existing features ---
    features['coord_std_dev_x'] = coords[:, 0].std()
    features['coord_std_dev_y'] = coords[:, 1].std()
    depot_coord, customer_coords = coords[0], coords[1:]
    if len(customer_coords) > 0:
        dists_from_depot = np.linalg.norm(customer_coords - depot_coord, axis=1)
        features['avg_dist_from_depot'] = dists_from_depot.mean()
        features['max_dist_from_depot'] = dists_from_depot.max()
    else:
        features['avg_dist_from_depot'] = 0.0
        features['max_dist_from_depot'] = 0.0
        
    return features, mst_length