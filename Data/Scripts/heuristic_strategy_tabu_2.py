# heuristic_strategy_tabu_2.py
import os
import time
import copy
import math
import numpy as np
import collections
import pandas as pd
import matplotlib.pyplot as plt
import gc
import random
from typing import Dict, List, Any, Optional, Tuple
from itertools import combinations, product
from math import comb
import functools
from scipy.spatial import ConvexHull, Voronoi, QhullError
from vrp_utils import (
    parse_vrp, parse_sol, get_true_VRP_cost,
    write_solution_file, plot_solution,
    verify_solution_feasibility, estimate_tsp_tour_length
)
from initial_solution_generator import generate_feasible_solution_regret

class VRPInstanceSolverTabu2:
    """
    Solves a VRP instance using an advanced, three-campaign heuristic with memory.
    The search deterministically escalates through increasingly powerful campaigns,
    guided by a Tabu list and partial node locking to enhance efficiency.
    """
    def __init__(self, instance_path: str, solution_path: str, output_dir: str, ml_model: Any,
                 heuristic_timeout: int = 3600,
                 num_vehicles: Optional[int] = None, apply_offset: bool = True):
        
        self.instance_path = instance_path
        self.solution_path = solution_path
        self.output_dir = output_dir
        self.basename = os.path.splitext(os.path.basename(instance_path))[0]
        self.ml_model = ml_model
        self.heuristic_timeout = heuristic_timeout
        self.objective_function_mode = 'RegressionTree'
        self.num_vehicles = num_vehicles
        self.apply_offset = apply_offset
        # --- Core Heuristic State ---
        self.search_start_time = 0.0
        self.timing_results = {}
        self.estimator_gap_history = []
        self.true_cost_gap_history = []
        self.global_bests_for_lkh = []
        self.best_true_solution = None
        self.best_true_cost = float('inf')
        self.global_best_estimated_cost = float('inf')
        self.optimal_cost = 0
        self.optimal_estimated_cost = 0
        self.max_k_opt_level = 0
        self.timed_out = False
        
        # --- Memory and State Management ---
        self.current_iteration = 0
        self.tabu_list = {}
        self.tabu_tenure = 30 
        self.locked_nodes = set()

        # --- NEW: Heat and Perturbation Attributes ---
        self.heat = 0.0
        self.heat_factor = 0.005  # How much of a bad delta contributes to heat
        self.max_heat = 10000.0   # A cap to prevent runaway values
        self.heat_dissipation = 0.95 # Cooling factor per major cycle

    def _check_timeout(self):
        if self.timed_out: return True
        if time.time() - self.search_start_time > self.heuristic_timeout:
            self.timed_out = True
            return True
        return False

    def solve(self):
        overall_start_time = time.time()
        self._load_and_preprocess_data()
        
        self.timing_results['2. Heuristic Search'] = time.time()
        self._run_four_campaign_search()
        self.timing_results['2. Heuristic Search'] = time.time() - self.timing_results['2. Heuristic Search']
        
        self.timing_results['3. Final LKH Calculation'] = time.time()
        self._calculate_final_true_costs()
        self.timing_results['3. Final LKH Calculation'] = time.time() - self.timing_results['3. Final LKH Calculation']
        self._generate_reports(overall_start_time)
        gc.collect()

    def _load_and_preprocess_data(self):
        step_start_time = time.time()
        self.capacity, self.depot_id, self.coords, self.demands = parse_vrp(self.instance_path)
        self.depot_coord = self.coords[self.depot_id]
        self.all_customer_data_orig = {nid: {'coords': c, 'demand': self.demands.get(nid, 0)} for nid, c in self.coords.items()}
        self.customers = {nid: v for nid, v in self.all_customer_data_orig.items() if nid != self.depot_id}
        self.customer_list = sorted(list(self.customers.keys()))
        self.max_k_opt_level = len(self.customer_list) // 2
        self.tabu_tenure = min(30, len(self.customer_list) // 4)

        if os.path.exists(self.solution_path) and self.solution_path.endswith('.sol'):
            self.optimal_routes, self.optimal_cost = parse_sol(self.solution_path, apply_offset=self.apply_offset)
            if self.num_vehicles is None: self.num_vehicles = len(self.optimal_routes)
            self.optimal_estimated_cost = self._calculate_total_estimated_cost({i: r for i, r in enumerate(self.optimal_routes)})
        else:
            if self.num_vehicles is None: self.num_vehicles = 1
            self.optimal_routes, self.optimal_cost, self.optimal_estimated_cost = [], 0, 0
        
        self.timing_results['1. Data Loading'] = time.time() - step_start_time

    # --- Main Search Orchestrator ---
    
    def _run_four_campaign_search(self):
        self.search_start_time = time.time()
        self.timed_out = False

        initial_solution = generate_feasible_solution_regret(self.customers, self.depot_coord, self.num_vehicles, self.capacity)
        if not initial_solution:
            print(f"FATAL: Could not generate initial solution for {self.basename}. Aborting.")
            return

        current_solution = initial_solution
        self.global_best_estimated_cost = self._calculate_total_estimated_cost(current_solution)
        self._update_and_log_new_best(copy.deepcopy(current_solution), self.global_best_estimated_cost, 0.0)

        while not self._check_timeout():
            self.current_iteration += 1
            self._update_locked_nodes(current_solution)
            self.heat *= self.heat_dissipation # Apply cooling
            
            # --- CAMPAIGN 1: The Primary Improvement Engine ---
            # This campaign has its own internal loops and will run until it bottoms out.
            solution_after_C1, improved_C1 = self._run_campaign_1(current_solution)
            current_solution = solution_after_C1 # Always accept the state after the campaign

            # If the main engine made progress, restart the main loop to run it again.
            if improved_C1:
                continue
            # --- DIVERSIFICATION HELPERS: Run only if Campaign 1 is stuck ---
            print("\n[!] Campaign 1 stalled. Escalating to helper campaigns...")

            # Helper 1: Polar Shape-Tuning
            solution_after_C2, improved_C2 = self._run_polar_shape_tuning_campaign(current_solution)
            current_solution = solution_after_C2
            if improved_C2:
                print("[!] Campaign 2 succeeded. Returning to main engine.")
                continue # Improvement found, go back to Campaign 1

            # Helper 2: Constrained Regional Search
            solution_after_C3, improved_C3 = self._run_campaign_3(current_solution)
            current_solution = solution_after_C3
            if improved_C3:
                print("[!] Campaign 3 succeeded. Returning to main engine.")
                continue # Improvement found, go back to Campaign 1

            # Helper 3: Global 3-Opt and Perturbation
            solution_after_C4, improved_C4, perturbed = self._run_campaign_4(current_solution)
            current_solution = solution_after_C4
            if improved_C4 or perturbed:
                if perturbed:
                    self.heat = 0.0 # Reset heat after a perturbation
                print("[!] Campaign 4 succeeded. Returning to main engine.")
                continue # Improvement found, go back to Campaign 1

            # If we reach here, the engine is stalled and no helpers could improve the solution.
            print("\n[!!!] All campaigns are exhausted. Terminating search.")
            break

    # --- Campaign 1: Systematic Local Search ---

    def _run_campaign_1(self, solution):
        """
        Campaign 1: Systematic Local Search
        Sequence:
          1. Greedy supernode search (iterative worst anchors, full cross-vehicle scan, symmetry guard, locking)
          2. Probabilistic 1-opt polish (strict estimator gate, locking, tabu)
          3. Targeted one-for-one swaps (pressure ordering, symmetry guard, locking)
          4. Escalating exhaustive k-opt (3..max_local_k-1), first improvement only
        Exits immediately on first improvement in any tier.
        Updates global best only if the improvement beats it.
        """
        print("\n--- Campaign 1: Systematic Local Search ---")
        current_solution = copy.deepcopy(solution)
        start_cost = self._calculate_total_estimated_cost(current_solution)
        max_local_k = 4  # keep local k small for speed

        while not self._check_timeout():
            # --- Tier 1a: Supernode ---
            solution_after_super, improved_super = self._run_greedy_supernode_search(current_solution)
            if improved_super:
                new_cost = self._calculate_total_estimated_cost(solution_after_super)
                if new_cost < self.global_best_estimated_cost - 1e-9:
                    self._log_and_update_best(copy.deepcopy(solution_after_super), 'C1_supernode')
                return solution_after_super, new_cost < start_cost - 1e-9

            # --- Tier 1b: Probabilistic 1-opt polish ---
            solution_after_polish, improved_polish = self._run_probabilistic_greedy_polish(current_solution)
            if improved_polish:
                new_cost = self._calculate_total_estimated_cost(solution_after_polish)
                if new_cost < self.global_best_estimated_cost - 1e-9:
                    self._log_and_update_best(copy.deepcopy(solution_after_polish), 'C1_1-opt_polish')
                return solution_after_polish, new_cost < start_cost - 1e-9

            # --- Tier 1c: Targeted worst-node swaps ---
            solution_after_swaps, improved_swaps = self._run_targeted_one_for_one_swaps(current_solution)
            if improved_swaps:
                new_cost = self._calculate_total_estimated_cost(solution_after_swaps)
                if new_cost < self.global_best_estimated_cost - 1e-9:
                    self._log_and_update_best(copy.deepcopy(solution_after_swaps), 'C1_targeted_swap')
                return solution_after_swaps, new_cost < start_cost - 1e-9

            # --- Tier 2: Escalating exhaustive k-opt ---
            for k in range(3, max_local_k):
                if self._check_timeout():
                    break
                print(f"  > Searching exhaustive first-improvement {k}-opt...")
                current_solution_after_k, improved_k = self._run_exhaustive_first_improvement_k_opt(current_solution, k)
                if improved_k:
                    new_cost = self._calculate_total_estimated_cost(current_solution_after_k)
                    if new_cost < self.global_best_estimated_cost - 1e-9:
                        self._log_and_update_best(copy.deepcopy(current_solution_after_k), f'C1_{k}-opt')
                    return current_solution_after_k, new_cost < start_cost - 1e-9

            # No improvement found in any tier
            print(" > Campaign 1 has bottomed out.")
            return current_solution, False

        # Timeout reached without improvement
        return current_solution, False

    def _run_exhaustive_first_improvement_k_opt(self, solution, k):
        """
        Campaign 1 greedy 3-cycle swap:
        - Ignores passed k, always runs k=3 cyclic swap (v1->v2, v2->v3, v3->v1).
        - Uses marginal contribution scoring to pick candidates.
        - Top 1/3 worst nodes per route, capped at 10.
        - Force-unlock worst node in route if all nodes locked.
        - Shorter tabu tenure for this operator.
        - First improvement only, stops immediately once found.
        """
        current_solution = copy.deepcopy(solution)
        vehicle_ids = [vid for vid, nodes in current_solution.items() if nodes]
        if len(vehicle_ids) < 3:
            return current_solution, False

        local_tabu_tenure = max(8, self.tabu_tenure // 2)

        # ---- Step 1: precompute marginal contributions per node ----
        node_contrib = {}
        for vid in vehicle_ids:
            route_nodes = current_solution[vid]
            if not route_nodes:
                continue
            base_cost = self._get_route_cost(tuple(sorted(route_nodes)))
            for n in route_nodes:
                reduced_cost = self._get_route_cost(tuple(sorted(x for x in route_nodes if x != n)))
                node_contrib[(vid, n)] = base_cost - reduced_cost

        # ---- Step 2: select candidate nodes per route ----
        per_route_candidates = {}
        for vid in vehicle_ids:
            route_nodes = current_solution[vid]
            unlocked = [n for n in route_nodes if (n, vid) not in self.locked_nodes]

            if not unlocked and route_nodes:
                # force-include the single worst node
                worst_node = max(route_nodes, key=lambda n: node_contrib[(vid, n)])
                unlocked = [worst_node]

            contribs = [(n, node_contrib[(vid, n)]) for n in unlocked]
            contribs.sort(key=lambda x: x[1], reverse=True)

            n_to_try = min(max(1, len(unlocked) // 3), 10)
            per_route_candidates[vid] = [n for n, _ in contribs[:n_to_try]]

        # ---- Step 3: order routes by pressure ----
        route_pressure = []
        for vid, cand_nodes in per_route_candidates.items():
            contrib_vals = [node_contrib[(vid, n)] for n in cand_nodes[:3]] if cand_nodes else []
            route_pressure.append((vid, sum(contrib_vals)))
        route_pressure.sort(key=lambda x: x[1], reverse=True)

        # ---- Step 4: iterate triples ----
        for v_combo in combinations([vid for vid, _ in route_pressure], 3):
            c_lists = [per_route_candidates[vid] for vid in v_combo]
            if any(not cl for cl in c_lists):
                continue

            # sort candidate triples by sum of marginal contributions
            triple_list = [((c_a, c_b, c_c),
                            node_contrib[(v_combo[0], c_a)] +
                            node_contrib[(v_combo[1], c_b)] +
                            node_contrib[(v_combo[2], c_c)])
                           for c_a in c_lists[0] for c_b in c_lists[1] for c_c in c_lists[2]]
            triple_list.sort(key=lambda x: x[1], reverse=True)

            # ---- Step 5: try triples ----
            for c_combo, _score in triple_list:
                from_v = v_combo
                to_v = (v_combo[1], v_combo[2], v_combo[0])  # cyclic shift

                # capacity check
                demands = [self.customers[c]['demand'] for c in c_combo]
                loads = {vid: sum(self.customers[n]['demand'] for n in current_solution[vid]) for vid in v_combo}
                temp_loads = loads.copy()
                for i in range(3):
                    temp_loads[from_v[i]] -= demands[i]
                    temp_loads[to_v[i]] += demands[i]
                if any(l > self.capacity for l in temp_loads.values()):
                    continue

                move_dict = {'type': 'C1_3-opt', 'customers': c_combo, 'from_v': from_v, 'to_v': to_v}
                move_tuple = self._get_canonical_move(move_dict)
                delta = self._calculate_k_opt_delta(c_combo, from_v, to_v, current_solution)

                # aspiration: allow tabu if move improves global best
                if self._is_tabu(move_tuple) and not (delta < self.global_best_estimated_cost - 1e-9):
                    continue

                if delta < -1e-9:
                    print(f"    > Found improving greedy 3-swap (delta: {delta:+.2f})")
                    new_solution = self._apply_move_to_copy(current_solution, move_dict)
                    # Apply shorter tenure manually
                    self.tabu_list[move_tuple] = self.current_iteration + local_tabu_tenure
                    self._log_and_update_best(new_solution, 'C1_3-opt')
                    self._rebuild_locked_nodes(new_solution)
                    return new_solution, True

        return current_solution, False


    # --- Campaign 2: Polar Shape-Tuning --- 
    def _run_polar_shape_tuning_campaign(self, solution):
        print("\n--- Campaign 2: Polar Shape-Tuning ---")
        current_solution = copy.deepcopy(solution)
        improvement_found = False

        # Pre-calculate polar coordinates for all customers relative to the depot
        node_to_vehicle = self._get_node_to_vehicle_map(current_solution)
        polar_coords = {}
        for nid, data in self.customers.items():
            dx = data['coords'][0] - self.depot_coord[0]
            dy = data['coords'][1] - self.depot_coord[1]
            polar_coords[nid] = {
                'rho': np.sqrt(dx**2 + dy**2),      # Radius / Distance
                'theta': np.arctan2(dy, dx)       # Angle
            }

        # --- Angular Optimization (Un-twisting routes) ---
        angular_improvement = True
        while angular_improvement:
            if self._check_timeout(): break
            angular_improvement = False
            sorted_by_angle = sorted(self.customer_list, key=lambda c: polar_coords[c]['theta'])

            # Include wrap-around for circular continuity
            pairs = [(sorted_by_angle[i], sorted_by_angle[i+1]) for i in range(len(sorted_by_angle)-1)]
            pairs.append((sorted_by_angle[-1], sorted_by_angle[0]))

            for c1, c2 in pairs:
                v1, v2 = node_to_vehicle.get(c1), node_to_vehicle.get(c2)
                if v1 is None or v2 is None or v1 == v2: continue

                nodes_v1 = [c for c in current_solution[v1] if (c, v1) not in self.locked_nodes]
                nodes_v2 = [c for c in current_solution[v2] if (c, v2) not in self.locked_nodes]

                nodes_v1.sort(key=lambda c: polar_coords[c]['theta'])
                nodes_v2.sort(key=lambda c: polar_coords[c]['theta'])

                try:
                    i1 = nodes_v1.index(c1)
                    i2 = nodes_v2.index(c2)
                    # Symmetric window slicing for balanced border exploration
                    window_v1 = nodes_v1[max(0, i1-1): min(len(nodes_v1), i1+2)]
                    window_v2 = nodes_v2[max(0, i2-1): min(len(nodes_v2), i2+2)]

                    # Deduplicate while preserving order
                    seen = set()
                    window_nodes = [n for n in window_v1 + window_v2 if not (n in seen or seen.add(n))]
                    if len(window_nodes) < 2: continue
                except ValueError:
                    continue  # Node not in unlocked list

                new_solution_after_opt, improved_in_window = self._optimize_border_window(
                    current_solution, window_nodes, [v1, v2], 'C2_polar_angular'
                )
                if improved_in_window:
                    current_solution = new_solution_after_opt
                    node_to_vehicle = self._get_node_to_vehicle_map(current_solution)
                    improvement_found = angular_improvement = True
                    break  # Restart scan with updated solution

        # --- Radial Optimization (Reducing spread relative to depot) ---
        radial_improvement = True
        while radial_improvement:
            if self._check_timeout(): break
            radial_improvement = False
            attempted_pairs = set()  # Reset symmetry guard per full pass to avoid over-accumulation

            # Recompute route centroids and polar info
            route_info = {}
            polar_coords_centroids = {}
            for vid, nodes in current_solution.items():
                if not nodes: continue
                coords = np.array([self.customers[n]['coords'] for n in nodes])
                centroid = np.mean(coords, axis=0)
                route_info[vid] = {'centroid': centroid, 'nodes': nodes}
                dx = centroid[0] - self.depot_coord[0]
                dy = centroid[1] - self.depot_coord[1]
                polar_coords_centroids[vid] = {'theta': np.arctan2(dy, dx)}

            # Sort vehicles by centroid theta for systematic iteration
            sorted_vids = sorted(route_info.keys(), key=lambda vid: polar_coords_centroids[vid]['theta'])

            for v_main in sorted_vids:
                if self._check_timeout(): break
                main_centroid = route_info[v_main]['centroid']
                unlocked_main = [n for n in route_info[v_main]['nodes'] if (n, v_main) not in self.locked_nodes]

                # Force-unlock the overall farthest node from centroid if locked
                all_main_coords = np.array([self.customers[n]['coords'] for n in route_info[v_main]['nodes']])
                dists_all = np.linalg.norm(all_main_coords - main_centroid, axis=1)
                worst_dist_idx = np.argmax(dists_all)
                worst_dist_node = route_info[v_main]['nodes'][worst_dist_idx]
                if (worst_dist_node, v_main) in self.locked_nodes:
                    unlocked_main.append(worst_dist_node)

                if len(unlocked_main) < 1: continue

                # Select bad nodes: farthest from centroid among (possibly extended) unlocked
                main_coords = np.array([self.customers[n]['coords'] for n in unlocked_main])
                dists = np.linalg.norm(main_coords - main_centroid, axis=1)
                sorted_indices = np.argsort(dists)[::-1]  # Descending distance
                n_to_try = max(1, len(unlocked_main) // 3)
                bad_nodes = [unlocked_main[i] for i in sorted_indices[:n_to_try]]

                # Target vehicles sorted by centroid proximity (nearest first)
                targets = [(vid, np.linalg.norm(route_info[vid]['centroid'] - main_centroid))
                           for vid in route_info if vid != v_main]
                targets.sort(key=lambda x: x[1])

                load_main = sum(self.customers[n]['demand'] for n in current_solution[v_main])
                for bad_node in bad_nodes:
                    if self._check_timeout(): break
                    bad_demand = self.customers[bad_node]['demand']

                    for v_target, _ in targets:
                        if self._check_timeout(): break
                        load_target = sum(self.customers[n]['demand'] for n in current_solution[v_target])
                        unlocked_target = [n for n in route_info[v_target]['nodes'] if (n, v_target) not in self.locked_nodes]
                        if not unlocked_target: continue

                        # Candidates: closest to main_centroid
                        target_coords = np.array([self.customers[n]['coords'] for n in unlocked_target])
                        cand_dists = np.linalg.norm(target_coords - main_centroid, axis=1)
                        sorted_cand_indices = np.argsort(cand_dists)  # Ascending
                        m_to_try = max(1, len(unlocked_target) // 3)
                        candidates = [unlocked_target[i] for i in sorted_cand_indices[:m_to_try]]

                        for cand_node in candidates:
                            pair_key = frozenset({(v_main, bad_node), (v_target, cand_node)})
                            if pair_key in attempted_pairs: continue
                            attempted_pairs.add(pair_key)

                            cand_demand = self.customers[cand_node]['demand']
                            new_load_main = load_main - bad_demand + cand_demand
                            new_load_target = load_target - cand_demand + bad_demand
                            if new_load_main > self.capacity or new_load_target > self.capacity: continue

                            move_dict = {'type': 'swap', 'customers': (bad_node, cand_node), 'vehicles': (v_main, v_target)}
                            move_tuple = self._get_canonical_move(move_dict)
                            if self._is_tabu(move_tuple): continue

                            delta = self._calculate_swap_delta(bad_node, cand_node, v_main, v_target, current_solution)
                            if delta < -1e-9:
                                print(f"    > Found improving radial swap (delta: {delta:+.2f})")
                                new_solution = self._apply_move_to_copy(current_solution, move_dict)
                                self._add_to_tabu(move_tuple)
                                self._log_and_update_best(new_solution, 'C2_polar_radial')
                                current_solution = new_solution
                                node_to_vehicle = self._get_node_to_vehicle_map(current_solution)
                                improvement_found = radial_improvement = True
                                break  # First improvement
                        if radial_improvement: break
                    if radial_improvement: break
                if radial_improvement: break
        print(" > Campaign 2 Radial Tuning Failed to Find a Better Solution")
        return current_solution, improvement_found

    def _optimize_border_window(self, solution, window_nodes, affected_vehicles, event_type):
        """
        Performs an exhaustive search on a small set of nodes between specific vehicles.
        """
        node_to_vehicle = self._get_node_to_vehicle_map(solution)
        
        # Generate all possible re-assignments of window_nodes to the affected_vehicles
        for assignment in product(affected_vehicles, repeat=len(window_nodes)):
            if self._check_timeout(): break
            
            # Create the move dictionary for this potential reassignment
            move_dict = {
                'type': f'{len(window_nodes)}-opt',
                'customers': tuple(window_nodes),
                'from_v': tuple(node_to_vehicle[c] for c in window_nodes),
                'to_v': assignment
            }

            # Check if it's a real move and not tabu
            if move_dict['from_v'] == move_dict['to_v']: continue
            move_tuple = self._get_canonical_move(move_dict)
            if self._is_tabu(move_tuple): continue

            # --- START OF FIX ---
            # Explicitly pass only the arguments that the helper functions expect.
            temp_sol = self._apply_k_opt_to_temp(solution, 
                                                 customers=move_dict['customers'], 
                                                 from_v=move_dict['from_v'], 
                                                 to_v=move_dict['to_v'])
            if not self._is_feasible(temp_sol, affected_vehicles): continue

            # Evaluate delta and apply if improving
            delta = self._calculate_k_opt_delta(customers=move_dict['customers'], 
                                                from_v=move_dict['from_v'], 
                                                to_v=move_dict['to_v'], 
                                                solution=solution)
            # --- END OF FIX ---

            if delta < -1e-9:
                new_solution = self._apply_move_to_copy(solution, move_dict)
                self._add_to_tabu(move_tuple)
                self._log_and_update_best(new_solution, event_type)
                return new_solution, True
                
        return solution, False
        
    # --- Campaign 3: Constrained Regional Search ---

    def _run_campaign_3(self, solution):
        """
        Campaign 3: Fine-Grained Border Refinement (Convex Hull + Voronoi Guard + kNN Expansion)

        Improvements over centroid-only version:
          - Convex hull vertices as seeds (captures outer shape).
          - Voronoi guard: nodes near equipotential line between centroids.
          - kNN expansion with separate radius parameter (not reusing tau).
          - Normalized |Δφ| values, stable across scales.
          - Vectorized distance computations for efficiency.
          - Safer set-based merging and mapping.

        Determinism & bounds:
          - Scan up to max_vehicles worst-cost routes.
          - Build bounded windows (<= max_window_nodes).
          - Try swaps first, then relocates, return on first improvement.
        """
        print("\n--- Campaign 3: Fine-Grained Border Refinement ---")
        current_solution = copy.deepcopy(solution)

        # --- Parameters (tuneable) ---
        max_window_nodes = 12
        knn_k = 3
        voronoi_tau_scale = 0.10    # fraction of inter-centroid distance
        knn_radius_scale = 0.05     # fraction of inter-centroid distance
        attempt_relocate = True

        # --- Route info: centroids, costs, nodes ---
        route_info = {}
        for vid, nodes in current_solution.items():
            if not nodes:
                continue
            coords = np.array([self.customers[n]['coords'] for n in nodes])
            route_info[vid] = {
                'centroid': np.mean(coords, axis=0),
                'cost': self._get_route_cost(tuple(sorted(nodes))),
                'nodes': nodes
            }
        if len(route_info) < 2:
            print(" > Campaign 3 skipped: fewer than 2 non-empty vehicles.")
            return current_solution, False

        # Worst-first vehicles
        selected_vids = list(route_info.keys())

        if len(selected_vids) < 2:
            return current_solution, False

        # Precompute loads for capacity checks
        route_loads = {vid: sum(self.customers[n]['demand']
                                for n in current_solution.get(vid, []))
                       for vid in selected_vids}

        # Precompute convex hulls
        hulls = self._compute_convex_hull_nodes(current_solution, selected_vids)

        # Iterate vehicle pairs
        for i_idx, v_a in enumerate(selected_vids):
            if self._check_timeout():
                break
            c_a = route_info[v_a]['centroid']

            neighbors = sorted([v_b for v_b in selected_vids if v_b != v_a],
                               key=lambda v_b: np.linalg.norm(c_a - route_info[v_b]['centroid']))

            for v_b in neighbors:
                if self._check_timeout():
                    break
                c_b = route_info[v_b]['centroid']
                seg_len = np.linalg.norm(c_a - c_b)
                if seg_len < 1e-9:
                    continue

                voronoi_tau = voronoi_tau_scale * seg_len
                knn_radius = max(1e-6, knn_radius_scale * seg_len)

                # Collect border window
                window_nodes, window_map = self._collect_pair_border_window(
                    route_info, hulls, v_a, v_b, seg_len,
                    voronoi_tau, knn_radius, knn_k, max_window_nodes
                )
                if len(window_nodes) < 2 or len({window_map[n] for n in window_nodes}) < 2:
                    continue

                # --- Try swaps ---
                attempted_pairs = set()
                for idx_i, n_i in enumerate(window_nodes):
                    from_i = window_map[n_i]
                    for n_j in window_nodes[idx_i + 1:]:
                        to_j = window_map[n_j]
                        if from_i == to_j:
                            continue

                        pair_key = frozenset({(from_i, n_i), (to_j, n_j)})
                        if pair_key in attempted_pairs:
                            continue
                        attempted_pairs.add(pair_key)

                        load_i_after = route_loads[from_i] - self.customers[n_i]['demand'] + self.customers[n_j]['demand']
                        load_j_after = route_loads[to_j] - self.customers[n_j]['demand'] + self.customers[n_i]['demand']
                        if load_i_after > self.capacity or load_j_after > self.capacity:
                            continue

                        move_dict = {'type': 'swap', 'customers': (n_i, n_j), 'vehicles': (from_i, to_j)}
                        move_tuple = self._get_canonical_move(move_dict)
                        if self._is_tabu(move_tuple):
                            continue

                        delta = self._calculate_swap_delta(n_i, n_j, from_i, to_j, current_solution)
                        if delta < -1e-9:
                            print(f"    > C3 border swap {from_i}↔{to_j} (Δ {delta:+.3f})")
                            new_solution = self._apply_move_to_copy(current_solution, move_dict)
                            self._add_to_tabu(move_tuple)
                            self._log_and_update_best(new_solution, 'C3_border_swap')
                            return new_solution, True

                # --- Try relocates if swaps fail ---
                if attempt_relocate:
                    for n_i in window_nodes:
                        from_i = window_map[n_i]
                        for to_v in [v_a, v_b]:
                            if to_v == from_i:
                                continue
                            if route_loads[to_v] + self.customers[n_i]['demand'] > self.capacity:
                                continue

                            move_dict = {'type': 'relocate', 'customer': n_i, 'from': from_i, 'to': to_v}
                            move_tuple = self._get_canonical_move(move_dict)
                            if self._is_tabu(move_tuple):
                                continue

                            delta = self._calculate_relocate_delta(n_i, from_i, to_v, current_solution)
                            if delta < -1e-9:
                                print(f"    > C3 border relocate {from_i}→{to_v} (Δ {delta:+.3f})")
                                new_solution = self._apply_move_to_copy(current_solution, move_dict)
                                self._add_to_tabu(move_tuple)
                                self._log_and_update_best(new_solution, 'C3_border_reloc')
                                return new_solution, True

        print(" > Campaign 3 border refinement found no improving moves.")
        return current_solution, False


    def _collect_pair_border_window(self, route_info, hulls, v_a, v_b,
                                    seg_len, voronoi_tau, knn_radius,
                                    knn_k, max_window_nodes):
        """
        Build border window for vehicle pair (v_a, v_b):
          - Hull seeds (from scipy ConvexHull)
          - Voronoi guard (|Δφ| <= voronoi_tau)
          - kNN expansion within knn_radius
        """
        c_a, c_b = route_info[v_a]['centroid'], route_info[v_b]['centroid']
        nodes_a, nodes_b = route_info[v_a]['nodes'], route_info[v_b]['nodes']
        nodes_a_set, nodes_b_set = set(nodes_a), set(nodes_b)
        hull_a, hull_b = set(hulls.get(v_a, [])), set(hulls.get(v_b, []))

        # --- Vectorized Voronoi diffs ---
        coords_a = np.array([self.customers[n]['coords'] for n in nodes_a]) if nodes_a else np.empty((0, 2))
        coords_b = np.array([self.customers[n]['coords'] for n in nodes_b]) if nodes_b else np.empty((0, 2))

        dphi_a, dphi_b = [], []
        if coords_a.size > 0:
            dist_a_to_ca = np.linalg.norm(coords_a - c_a[None, :], axis=1)
            dist_a_to_cb = np.linalg.norm(coords_a - c_b[None, :], axis=1)
            dphi_a = np.abs(dist_a_to_cb - dist_a_to_ca) / seg_len
        if coords_b.size > 0:
            dist_b_to_ca = np.linalg.norm(coords_b - c_a[None, :], axis=1)
            dist_b_to_cb = np.linalg.norm(coords_b - c_b[None, :], axis=1)
            dphi_b = np.abs(dist_b_to_ca - dist_b_to_cb) / seg_len

        dphi_lookup = {}
        candidates_a, candidates_b = [], []

        for idx, n in enumerate(nodes_a):
            dphi = dphi_a[idx]
            dphi_lookup[n] = dphi
            is_hull = n in hull_a
            near_voronoi = dphi <= voronoi_tau / seg_len
            knn_adjacent = False
            if coords_b.size > 0:
                xi = np.array(self.customers[n]['coords'])
                dists = np.linalg.norm(coords_b - xi, axis=1)
                if dists.size > 0 and np.min(dists) <= knn_radius:
                    knn_adjacent = True
            if is_hull or near_voronoi or knn_adjacent:
                candidates_a.append((n, dphi))

        for idx, n in enumerate(nodes_b):
            dphi = dphi_b[idx]
            dphi_lookup[n] = dphi
            is_hull = n in hull_b
            near_voronoi = dphi <= voronoi_tau / seg_len
            knn_adjacent = False
            if coords_a.size > 0:
                xi = np.array(self.customers[n]['coords'])
                dists = np.linalg.norm(coords_a - xi, axis=1)
                if dists.size > 0 and np.min(dists) <= knn_radius:
                    knn_adjacent = True
            if is_hull or near_voronoi or knn_adjacent:
                candidates_b.append((n, dphi))

        # --- Keep at most 1/3 of candidates, capped at 10 ---
        def keep_closest_third(cands):
            if not cands:
                return []
            cands_sorted = sorted(cands, key=lambda t: t[1])
            k = max(1, int(math.ceil(len(cands_sorted) / 3)))
            k = min(k, 10)
            return [n for n, _ in cands_sorted[:k]]

        seeds_a = keep_closest_third(candidates_a)
        seeds_b = keep_closest_third(candidates_b)

        # Expand with kNN from opposite vehicle
        expanded_a = set(seeds_a)
        for n in seeds_a:
            expanded_a.update(self._k_nearest_from_vehicle(n, v_b, route_info, k=knn_k))
        expanded_b = set(seeds_b)
        for n in seeds_b:
            expanded_b.update(self._k_nearest_from_vehicle(n, v_a, route_info, k=knn_k))

        # Merge & dedup
        merged = list(expanded_a | expanded_b)
        merged_scored = [(nid, dphi_lookup.get(nid, 1.0),
                          v_a if nid in nodes_a_set else v_b) for nid in merged]
        merged_scored.sort(key=lambda t: t[1])

        window_nodes = [nid for nid, _, _ in merged_scored[:max_window_nodes]]
        window_map = {nid: (v_a if nid in nodes_a_set else v_b) for nid in window_nodes}

        # Ensure both vehicles represented
        vids_present = {window_map[n] for n in window_nodes}
        if vids_present != {v_a, v_b}:
            if v_a not in vids_present and seeds_a:
                for nid in seeds_a[:2]:
                    if nid not in window_map and len(window_nodes) < max_window_nodes:
                        window_nodes.append(nid)
                        window_map[nid] = v_a
            if v_b not in vids_present and seeds_b:
                for nid in seeds_b[:2]:
                    if nid not in window_map and len(window_nodes) < max_window_nodes:
                        window_nodes.append(nid)
                        window_map[nid] = v_b

        return window_nodes, window_map

    def _k_nearest_from_vehicle(self, node_id, vehicle_id, route_info, k=3):
        """Return up to k nearest nodes from vehicle_id to node_id (vectorized)."""
        target_nodes = route_info[vehicle_id]['nodes']
        if not target_nodes:
            return []
        xi = np.array(self.customers[node_id]['coords'])
        coords = np.array([self.customers[n]['coords'] for n in target_nodes])
        dists = np.linalg.norm(coords - xi[None, :], axis=1)
        idx_sorted = np.argsort(dists)
        nearest = []
        for idx in idx_sorted:
            nid = target_nodes[idx]
            if nid == node_id:
                continue
            nearest.append(nid)
            if len(nearest) >= k:
                break
        return nearest

    def _compute_convex_hull_nodes(self, solution, vehicle_ids):
        hulls = {}
        for vid in vehicle_ids:
            nodes = solution.get(vid, [])
            if len(nodes) < 3:
                hulls[vid] = nodes[:]  # degenerate hull
                continue

            coords = np.array([self.customers[n]['coords'] for n in nodes])
            try:
                hull = ConvexHull(coords)
                hull_indices = list(dict.fromkeys(hull.vertices))
                hull_nodes = [nodes[i] for i in hull_indices]
                hulls[vid] = hull_nodes
            except Exception as e:
                # QhullError or others: fallback to returning outermost two or all nodes
                # choose extreme nodes by angle or distance as a robust fallback:
                center = coords.mean(axis=0)
                angles = np.arctan2(coords[:,1]-center[1], coords[:,0]-center[0])
                order = np.argsort(angles)
                hulls[vid] = [nodes[i] for i in order]  # deterministic fallback
        return hulls

    # --- Campaign 4: Prioritized Global Search ---

    def _run_campaign_4(self, solution):
        print("\n--- Campaign 4: 3-Opt with Heat-Triggered Perturbation ---")
        prioritized_customers = self._get_prioritized_customer_list(solution)
        vehicle_ids = sorted([vid for vid, nodes in solution.items() if nodes])
        
        if len(vehicle_ids) < 3 or len(prioritized_customers) < 3:
            return solution, False, False

        # Exhaustive, prioritized 3-opt search
        for c_combo in combinations(prioritized_customers, 3):
            if self._check_timeout(): break

            # --- Probabilistic Perturbation Check ---
            prob = self._calculate_perturb_probability(self.heat)
            if random.random() < prob:
                print(f"    [!] Heat level {self.heat:.2f} triggered perturbation (Prob: {prob:.2%})")
                perturbed_solution = self._run_ruin_and_recreate_perturbation(solution, self.heat)
                self._log_and_update_best(perturbed_solution, 'Perturbation')
                return perturbed_solution, True, True # (solution, improved, perturbed)

            # --- Systematic 3-Opt Move Evaluation ---
            node_to_vehicle = self._get_node_to_vehicle_map(solution)
            v_combo_for_nodes = tuple(node_to_vehicle.get(c) for c in c_combo)
            if None in v_combo_for_nodes or len(set(v_combo_for_nodes)) < 3:
                continue

            v1, v2, v3 = v_combo_for_nodes
            c1, c2, c3 = c_combo
            
            # Test cyclic swap
            from_v, to_v = (v1, v2, v3), (v2, v3, v1)
            move_dict = {'type': 'C4_3-opt', 'customers': c_combo, 'from_v': from_v, 'to_v': to_v}
            move_tuple = self._get_canonical_move(move_dict)
            if self._is_tabu(move_tuple): continue

            delta = self._calculate_k_opt_delta(c_combo, from_v, to_v, solution)
            if delta < -1e-9:
                print(f"    > Found improving global 3-opt move (delta: {delta:+.2f})")
                new_solution = self._apply_move_to_copy(solution, move_dict)
                self._add_to_tabu(move_tuple)
                self._log_and_update_best(new_solution, 'C4_3-opt')
                return new_solution, True, False
            elif delta > 0: # It's a non-improving move, add to heat
                self.heat = min(self.max_heat, self.heat + delta * self.heat_factor)
                
        print(" > Campaign 4 has bottomed out.")
        return solution, False, False

    def _get_prioritized_customer_list(self, solution):
        scores = {}
        all_coords = np.array([self.customers[c]['coords'] for c in self.customer_list])
        centroid = np.mean(all_coords, axis=0)

        for cust in self.customer_list:
            node_to_vehicle = self._get_node_to_vehicle_map(solution)
            vid = node_to_vehicle.get(cust)
            if not vid or (cust, vid) in self.locked_nodes: continue

            cost_contrib = self._calculate_customer_cost_contribution(solution, cust)
            dist_from_centroid = np.linalg.norm(np.array(self.customers[cust]['coords']) - centroid)
            
            # Normalize and weigh the scores (example weights)
            scores[cust] = (0.7 * cost_contrib) + (0.3 * dist_from_centroid)
            
        return sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

    def _calculate_customer_cost_contribution(self, solution, customer_id):
        node_to_vehicle = self._get_node_to_vehicle_map(solution)
        vid = node_to_vehicle.get(customer_id)
        if not vid: return 0.0
        
        route_nodes = solution[vid]
        # --- START OF FIX ---
        # Convert the route_nodes list to a sorted, hashable tuple before calling the cached function.
        cost_with = self._get_route_cost(tuple(sorted(route_nodes)))
        cost_without = self._get_route_cost(tuple(sorted([n for n in route_nodes if n != customer_id])))
        # --- END OF FIX ---
        return cost_with - cost_without

    def _find_improving_k_opt_move(self, k_level, solution, customer_subset):
        if len(customer_subset) < k_level: return None
        
        combo_gen = combinations(customer_subset, k_level)
        vehicle_ids = sorted(list(solution.keys()))
        
        for combo in combo_gen:
            if self._check_timeout(): return None
            move = self._process_combo(combo, solution, vehicle_ids)
            if move:
                print(f"    > Found improving global {k_level}-opt move (delta: {move['cost_delta']:+.2f})")
                return move
        return None
    
    def _process_combo(self, cust_combo, solution, vehicle_ids):
        node_to_vehicle = self._get_node_to_vehicle_map(solution)
        from_v = tuple(node_to_vehicle.get(c) for c in cust_combo if node_to_vehicle.get(c) is not None)
        # Ensure combo is still valid if a node somehow wasn't in the map
        if len(from_v) != len(cust_combo): return None

        for to_v in product(vehicle_ids, repeat=len(cust_combo)):
            if self._check_timeout(): return None
            if from_v == to_v: continue

            # --- START OF FIX ---
            # Use separate variables for the dictionary (mutable) and the tuple (for tabu checks)
            move_dict = {'type': f'{len(cust_combo)}-opt', 'customers': cust_combo, 'from_v': from_v, 'to_v': to_v}
            move_tuple = self._get_canonical_move(move_dict)
            if self._is_tabu(move_tuple): continue
            # --- END OF FIX ---

            temp_solution = self._apply_k_opt_to_temp(solution, cust_combo, from_v, to_v)
            if not self._is_feasible(temp_solution, set(from_v) | set(to_v)):
                continue

            delta = self._calculate_k_opt_delta(cust_combo, from_v, to_v, solution)
            if delta < -1e-9:
                # Add the cost_delta to the dictionary and return the dictionary
                move_dict['cost_delta'] = delta
                return move_dict
        return None
        
    # --- Memory (Tabu/Locking) and Helper Methods ---

    def _get_canonical_move(self, move):
        """
        Canonicalize moves while preserving customer <-> vehicle mapping.
        Swap canonical form: ('swap', ((v1,c1),(v2,c2))) where the pairs are ordered by vehicle id.
        Relocate: ('relocate', customer, from, to)
        Opt: ('opt', tuple(sorted((from_v, customer) pairs)), tuple(sorted((to_v, customer) pairs)))
        """
        m_type = move.get('type', '')
        if 'relocate' in m_type:
            return ('relocate', move['customer'], move['from'], move['to'])
        elif 'swap' in m_type:
            # Expect exactly two customers and two vehicles
            c1, c2 = move['customers']
            v1, v2 = move['vehicles']
            # Keep the explicit mapping (vehicle,customer) and sort by vehicle id for canonical ordering.
            pair1 = (v1, c1)
            pair2 = (v2, c2)
            if pair1 <= pair2:
                return ('swap', (pair1, pair2))
            else:
                return ('swap', (pair2, pair1))
        elif 'opt' in m_type:
            # Build ordered pairs so mapping preserved and canonicalizable
            from_pairs = tuple(sorted((int(f), c) for f, c in zip(move['from_v'], move['customers'])))
            to_pairs = tuple(sorted((int(t), c) for t, c in zip(move['to_v'], move['customers'])))
            return ('opt', from_pairs, to_pairs)
        # Fallback: stable sorted tuple of items
        return tuple(sorted(move.items()))

    def _is_tabu(self, move_tuple):
        return move_tuple in self.tabu_list and self.tabu_list[move_tuple] > self.current_iteration

    def _add_to_tabu(self, move_tuple):
        self.tabu_list[move_tuple] = self.current_iteration + self.tabu_tenure
        # Prune old entries to prevent unbounded growth
        if len(self.tabu_list) > 20000:
            self.tabu_list = {m: exp for m, exp in self.tabu_list.items() if exp > self.current_iteration}

    def _update_locked_nodes(self, solution):
        # Periodically review locks to prevent stagnation
        if self.current_iteration % 50 == 0: 
            self._rebuild_locked_nodes(solution)

    def _rebuild_locked_nodes(self, solution):
        """Clears and rebuilds the locked_nodes set based on the top 25% of routes in the provided solution."""
        self.locked_nodes.clear()
    
        route_costs = {vid: self._get_route_cost(tuple(sorted(nodes))) for vid, nodes in solution.items() if nodes}
        if not route_costs: return
    
        # Lock nodes in the best 25% of routes
        sorted_routes = sorted(route_costs.keys(), key=lambda vid: route_costs[vid])
        num_to_lock = len(sorted_routes) // 4
    
        if num_to_lock > 0:
            for i in range(num_to_lock):
                vid = sorted_routes[i]
                for cust in solution[vid]:
                    self.locked_nodes.add((cust, vid))
            print(f"    [Memory] Locks reset. Locked {len(self.locked_nodes)} nodes in {num_to_lock} best routes.")

    def _log_and_update_best(self, new_solution, event_type):
        new_cost = self._calculate_total_estimated_cost(new_solution)
        self._log_gap_event(time.time() - self.search_start_time, new_cost, event_type)
        if new_cost < self.global_best_estimated_cost:
            self.global_best_estimated_cost = new_cost
            self._update_and_log_new_best(copy.deepcopy(new_solution), new_cost, time.time() - self.search_start_time)
        
            # --- NEW: Immediately reset locks based on the new best solution ---
            self._rebuild_locked_nodes(new_solution)
        
            return True
        return False

    # --- Calculation, Utility, and Reporting Methods (largely unchanged) ---
    def _run_targeted_one_for_one_swaps(self, solution, max_partners_per_route=32):
        """
        Progressive 1-for-1 targeted swaps with strict locking, tabu, and symmetry guard:
        - Precompute per-route unlocked nodes and their marginal contributions.
        - Order routes by 'pressure' (sum of top contributions) to focus on worst first.
        - For each worst node c in route v_a, try swaps with candidate nodes d in other routes v_b.
        - Capacity checked on both routes; tabu checked; symmetry guard prevents mirroring c↔d across the pass.
        - First-improvement acceptance: apply atomically, register tabu, rebuild locks, restart from top.
        - Return when a full pass yields no improvements.
        """
        current_solution = copy.deepcopy(solution)

        while not self._check_timeout():
            improvement_made = False
            attempted_pairs = set()  # symmetry guard across this pass

            # Precompute unlocked nodes per route
            per_route_unlocked = {
                vid: [n for n in current_solution.get(vid, []) if (n, vid) not in self.locked_nodes]
                for vid in current_solution
            }

            # Compute contributions for unlocked nodes per route
            per_route_contribs = {}
            for vid, nodes in per_route_unlocked.items():
                if not nodes:
                    per_route_contribs[vid] = []
                    continue
                route_cost_before = self._get_route_cost(tuple(sorted(nodes)))
                contribs = []
                for n in nodes:
                    # Marginal contribution = route cost - route cost without n
                    cost_without = self._get_route_cost(tuple(sorted([x for x in nodes if x != n])))
                    contribs.append((n, route_cost_before - cost_without))
                contribs.sort(key=lambda x: x[1], reverse=True)  # worst first
                per_route_contribs[vid] = contribs

            # Compute route pressure (sum of top 3 contributions)
            route_pressure = []
            for vid, contribs in per_route_contribs.items():
                pressure = sum(c for _, c in contribs[:min(3, len(contribs))])
                route_pressure.append((vid, pressure))
            route_pressure.sort(key=lambda x: x[1], reverse=True)

            # Iterate routes by descending pressure
            for v_a, _pressure in route_pressure:
                if self._check_timeout():
                    break
                worst_list_a = per_route_contribs.get(v_a, [])
                if not worst_list_a:
                    continue

                # Consider worst nodes in this route
                for c, _c_contrib in worst_list_a:
                    # Try candidates from other routes
                    for v_b in sorted(current_solution.keys()):
                        if v_b == v_a:
                            continue
                        partners_b = per_route_contribs.get(v_b, [])
                        if not partners_b:
                            continue

                        # Capacity snapshot
                        load_a = sum(self.customers[n]['demand'] for n in current_solution[v_a])
                        load_b = sum(self.customers[n]['demand'] for n in current_solution[v_b])

                        # Limit partner count for speed
                        for d, _d_contrib in partners_b[:max_partners_per_route]:
                            # Symmetry guard: skip attempted pair in this pass
                            pair_key = frozenset({(v_a, c), (v_b, d)})
                            if pair_key in attempted_pairs:
                                continue
                            attempted_pairs.add(pair_key)

                            # Capacity feasibility after swap
                            load_a_after = load_a - self.customers[c]['demand'] + self.customers[d]['demand']
                            load_b_after = load_b - self.customers[d]['demand'] + self.customers[c]['demand']
                            if load_a_after > self.capacity or load_b_after > self.capacity:
                                continue

                            move_dict = {'type': 'swap', 'customers': (c, d), 'vehicles': (v_a, v_b)}
                            move_tuple = self._get_canonical_move(move_dict)
                            if self._is_tabu(move_tuple):
                                continue

                            delta = self._calculate_swap_delta(c, d, v_a, v_b, current_solution)
                            if delta < -1e-9:
                                new_solution = self._apply_move_to_copy(current_solution, move_dict)
                                self._add_to_tabu(move_tuple)
                                self._log_and_update_best(new_solution, 'C1_targeted_swap')
                                self._rebuild_locked_nodes(new_solution)
                                current_solution = new_solution
                                improvement_made = True
                                break  # restart from top (first-improvement)
                        if improvement_made:
                            break
                    if improvement_made:
                        break

            if not improvement_made:
                # Full pass without improvement
                return current_solution, False

        # Timeout or loop exit without improvement
        return current_solution, False

    def _run_greedy_supernode_search(self, solution):
        """
            Supernode operator with full cross-vehicle worst-anchor coverage, strict locking, tabu, and symmetry guard:
            - For each vehicle v_main, take its top n_to_try worst nodes as anchors (unlocked only).
            - For each anchor_main, build supernode_main of size k from unlocked nodes (anchor + k-1 nearest).
            - For each target vehicle v_target != v_main:
                * Try relocate of supernode_main into v_target (capacity-checked, tabu-checked).
                * Try swaps with each of the n_to_try worst anchors in v_target (unlocked only).
        - Symmetry guard: skip if (v_target, anchor_target) ↔ (v_main, anchor_main) was already attempted this pass.
        - First-improvement acceptance: on improvement, apply atomically, register tabu, rebuild locks, return.
        - Iterate k from larger to smaller; exhaust all main anchors before shrinking k.
        """
        current_solution = copy.deepcopy(solution)
        attempted_pairs = set()  # symmetry guard for anchor pairs this pass

        while not self._check_timeout():
            improvement_found = False

            # Iterate vehicles deterministically to avoid noisy behavior
            for v_main in sorted(current_solution.keys()):
                # Unlocked nodes only for building anchors and supernodes
                unlocked_main = [n for n in current_solution.get(v_main, []) if (n, v_main) not in self.locked_nodes]
                if len(unlocked_main) < 2:
                    continue

                # Supernode size upper bound from route size
                route_len = len(unlocked_main)
                max_k = max(2, route_len // 3)
                n_to_try = max(1, route_len // 3)

                # Precompute per-route loads for quick capacity checks
                route_loads = {vid: sum(self.customers[n]['demand'] for n in nodes)
                               for vid, nodes in current_solution.items()}

                # Worst anchors for the main route (descending contribution)
                worst_main_anchors = self._find_n_worst_nodes_in_route(unlocked_main, n_to_try)

                # Descend k to exhaust stronger moves first
                for k in range(max_k, 1, -1):
                    if self._check_timeout():
                        break
                    if len(unlocked_main) < k:
                        continue

                    for anchor_main in worst_main_anchors:
                        # Build supernode_main from unlocked nodes
                        supernode_main = self._build_spatial_supernode(anchor_main, unlocked_main, k)
                        if len(supernode_main) < k:
                            continue
                        # Ensure all nodes in supernode are unlocked (belt-and-suspenders)
                        if any((c, v_main) in self.locked_nodes for c in supernode_main):
                            continue

                        demand_supernode_main = sum(self.customers[n]['demand'] for n in supernode_main)

                        # Try against every other vehicle as target
                        for v_target in sorted(current_solution.keys()):
                            if v_target == v_main:
                                continue

                            # ---------- Relocate attempt (main → target) ----------
                            if route_loads[v_target] + demand_supernode_main <= self.capacity:
                                move_dict = {
                                    'type': f'C1_SN_{k}-opt',
                                    'customers': tuple(supernode_main),
                                    'from_v': tuple([v_main] * k),
                                    'to_v': tuple([v_target] * k),
                                }
                                move_tuple = self._get_canonical_move(move_dict)
                                if not self._is_tabu(move_tuple):
                                    delta = self._calculate_supernode_relocate_delta(
                                        supernode_main, v_main, v_target, current_solution
                                    )
                                    if delta < -1e-9:
                                        new_solution = self._apply_move_to_copy(current_solution, move_dict)
                                        self._add_to_tabu(move_tuple)
                                        self._log_and_update_best(new_solution, f'C1_SN_Reloc_k{k}')
                                        self._rebuild_locked_nodes(new_solution)
                                        return new_solution, True  # first improvement

                            # ---------- Swap attempts with multiple target anchors ----------
                            unlocked_target = [n for n in current_solution.get(v_target, []) if (n, v_target) not in self.locked_nodes]
                            if len(unlocked_target) < k:
                                continue

                            worst_target_anchors = self._find_n_worst_nodes_in_route(
                                unlocked_target,
                                min(n_to_try, len(unlocked_target))
                            )

                            for anchor_target in worst_target_anchors:
                                # Symmetry guard: skip already-attempted anchor pair
                                pair_key = frozenset({(v_main, anchor_main), (v_target, anchor_target)})
                                if pair_key in attempted_pairs:
                                    continue
                                attempted_pairs.add(pair_key)

                                supernode_target = self._build_spatial_supernode(anchor_target, unlocked_target, k)
                                if len(supernode_target) < k:
                                    continue
                                if any((c, v_target) in self.locked_nodes for c in supernode_target):
                                    continue

                                demand_supernode_target = sum(self.customers[n]['demand'] for n in supernode_target)

                                load_main_after = route_loads[v_main] - demand_supernode_main + demand_supernode_target
                                load_target_after = route_loads[v_target] - demand_supernode_target + demand_supernode_main
                                if load_main_after > self.capacity or load_target_after > self.capacity:
                                    continue

                                # k1 and k2 are equal to k here but keep general form
                                k1, k2 = len(supernode_main), len(supernode_target)
                                customers = tuple(supernode_main + supernode_target)
                                from_v = tuple([v_main] * k1 + [v_target] * k2)
                                to_v = tuple([v_target] * k1 + [v_main] * k2)

                                move_dict = {
                                    'type': f'C1_SN_{k1+k2}-opt',
                                    'customers': customers,
                                    'from_v': from_v,
                                    'to_v': to_v,
                                }
                                move_tuple = self._get_canonical_move(move_dict)
                                if self._is_tabu(move_tuple):
                                    continue

                                delta = self._calculate_supernode_swap_delta(
                                    supernode_main, supernode_target, v_main, v_target, current_solution
                                )
                                if delta < -1e-9:
                                    new_solution = self._apply_move_to_copy(current_solution, move_dict)
                                    self._add_to_tabu(move_tuple)
                                    self._log_and_update_best(new_solution, f'C1_SN_Swap_k{k1}_{k2}')
                                    self._rebuild_locked_nodes(new_solution)
                                    return new_solution, True  # first improvement

            # If we reach here, no improvement in a full pass
            return current_solution, False
    
    def _find_n_worst_nodes_in_route(self, route_nodes: List[int], n: int) -> List[int]:
        """Finds the N nodes with the highest cost contribution in a given route."""
        if not route_nodes: return []

        contributions = []
        cost_before = self._get_route_cost(tuple(sorted(route_nodes)))
        for node in route_nodes:
            cost_after = self._get_route_cost(tuple(sorted([n for n in route_nodes if n != node])))
            contribution = cost_before - cost_after
            contributions.append((node, contribution))

        # Sort by contribution (descending) and return the top N nodes
        contributions.sort(key=lambda x: x[1], reverse=True)
        return [node for node, contrib in contributions[:n]]
    
    def _find_worst_node_in_route(self, route_nodes: List[int]) -> Optional[int]:
        """Finds the node with the highest cost contribution in a given route."""
        if not route_nodes: return None
        
        worst_node = None
        max_contribution = -float('inf')
        
        cost_before = self._get_route_cost(tuple(sorted(route_nodes)))
        for node in route_nodes:
            cost_after = self._get_route_cost(tuple(sorted([n for n in route_nodes if n != node])))
            contribution = cost_before - cost_after
            if contribution > max_contribution:
                max_contribution = contribution
                worst_node = node
        
        return worst_node

    def _build_spatial_supernode(self, anchor_node: int, available_nodes: List[int], k: int) -> List[int]:
        """Builds a supernode of size k using the anchor and its k-1 nearest neighbors."""
        if k <= 0: return []
        if len(available_nodes) < k: return available_nodes

        anchor_coords = np.array(self.customers[anchor_node]['coords'])
        
        # Calculate distance from the anchor to all other available nodes
        neighbors = []
        for node in available_nodes:
            if node == anchor_node: continue
            node_coords = np.array(self.customers[node]['coords'])
            dist = np.linalg.norm(anchor_coords - node_coords)
            neighbors.append((node, dist))
        
        # Sort neighbors by distance and take the k-1 closest
        neighbors.sort(key=lambda x: x[1])
        closest_neighbor_nodes = [n for n, d in neighbors[:k-1]]
        
        return [anchor_node] + closest_neighbor_nodes
        
    def _run_probabilistic_greedy_polish(self, solution):
        current_solution = copy.deepcopy(solution)
        improvement_made = False
        while not self._check_timeout():
            improving_moves = []
            node_to_vehicle = self._get_node_to_vehicle_map(current_solution)
            
            # Phase 1: Find all possible improving moves
            for c, from_vid in node_to_vehicle.items():
                if (c, from_vid) in self.locked_nodes: continue

                for to_vid in [k for k in current_solution if k != from_vid]:
                    if sum(self.customers[n]['demand'] for n in current_solution[to_vid]) + self.customers[c]['demand'] > self.capacity:
                        continue
                    
                    move_dict = {'type': 'relocate', 'customer': c, 'from': from_vid, 'to': to_vid}
                    move_tuple = self._get_canonical_move(move_dict)
                    if self._is_tabu(move_tuple): continue

                    delta = self._calculate_relocate_delta(c, from_vid, to_vid, current_solution)
                    if delta < -1e-9:
                        improving_moves.append({'move_dict': move_dict, 'move_tuple': move_tuple, 'delta': abs(delta)})
            
            if not improving_moves:
                break # No more improving moves found, exit the loop

            # Phase 2: Select one move probabilistically and apply it
            improvement_made = True
            total_delta = sum(move['delta'] for move in improving_moves)
            probabilities = [move['delta'] / total_delta for move in improving_moves]
            
            selected_index = np.random.choice(len(improving_moves), p=probabilities)
            best_move = improving_moves[selected_index]
            best_move_dict = best_move['move_dict']
            best_move_tuple = best_move['move_tuple']

            current_solution = self._apply_move_to_copy(current_solution, best_move_dict) # Pass the dictionary
            self._add_to_tabu(best_move_tuple)
            self._log_and_update_best(current_solution, 'C1_1-opt_polish')
        
        return current_solution, improvement_made

    def _is_feasible(self, solution, affected_vids):
        for vid in affected_vids:
            if sum(self.customers[n]['demand'] for n in solution.get(vid, [])) > self.capacity:
                return False
        return True

    def _apply_k_opt_to_temp(self, solution, customers, from_v, to_v):
        temp_sol = {vid: list(r) for vid, r in solution.items()}
        for i, cust in enumerate(customers):
            if cust in temp_sol[from_v[i]]: temp_sol[from_v[i]].remove(cust)
        for i, cust in enumerate(customers):
            temp_sol[to_v[i]].append(cust)
        return temp_sol
    
    def _calculate_perturb_probability(self, heat):
        """ Maps heat to a probability using a logistic function. """
        if heat <= 0: return 0.0
        # The probability starts rising around midpoint and gets steep
        midpoint = self.max_heat * 0.4
        steepness = 10 / self.max_heat
        prob = 1 / (1 + math.exp(-steepness * (heat - midpoint)))
        return prob
    
    def _run_ruin_and_recreate_perturbation(self, solution, heat):
        """ Strong destructive move that removes and reinserts a large portion of nodes. """
        perturbed_solution = copy.deepcopy(solution)
        all_nodes = [node for cluster in perturbed_solution.values() for node in cluster]
        if not all_nodes: return perturbed_solution
        destruction_level = self._calculate_perturb_probability(heat)*5 # Destroy Heated Amount of Solution
        num_to_remove = max(1, int(len(all_nodes) * destruction_level))
        
        # Ruin: Remove nodes, prioritizing non-locked ones
        unlocked_nodes = [n for n in all_nodes if not any((n, vid) in self.locked_nodes for vid in perturbed_solution)]
        nodes_to_remove = random.sample(unlocked_nodes, min(num_to_remove, len(unlocked_nodes)))
        node_to_key_map = self._get_node_to_vehicle_map(perturbed_solution)
        for node in set(nodes_to_remove):
            key = node_to_key_map.get(node)
            if key is not None and node in perturbed_solution[key]:
                perturbed_solution[key].remove(node)
        
        # Recreate: Greedily re-insert nodes
        nodes_to_reinsert = list(set(nodes_to_remove))
        random.shuffle(nodes_to_reinsert)

        for node in nodes_to_reinsert:
            best_vid, min_delta = -1, float('inf')
            
            # Find the cheapest place to insert the node
            for vid, route in perturbed_solution.items():
                if sum(self.customers[n]['demand'] for n in route) + self.customers[node]['demand'] <= self.capacity:
                    delta = self._calculate_relocate_delta(node, -1, vid, {**perturbed_solution, -1: []}) # Use a dummy 'from' vehicle
                    if delta < min_delta:
                        min_delta, best_vid = delta, vid
            
            if best_vid != -1:
                perturbed_solution[best_vid].append(node)
            else: # If no vehicle can take it, create a new route if possible
                new_vid = max(perturbed_solution.keys()) + 1 if perturbed_solution else 0
                perturbed_solution[new_vid] = [node]

        return {k:v for k,v in perturbed_solution.items() if v}
        
    @functools.lru_cache(maxsize=32768)
    def _get_route_cost(self, nodes_tuple: tuple) -> float:
        nodes_list = list(nodes_tuple)
        if not nodes_list: return 0.0
        nodes_in_route_coords = [self.customers[nid]['coords'] for nid in nodes_list] + [self.depot_coord]
        return estimate_tsp_tour_length(nodes_in_route_coords, mode=self.objective_function_mode, bounding_stats=self.ml_model)

    def _calculate_total_estimated_cost(self, solution):
        return sum(self._get_route_cost(tuple(sorted(nodes))) for nodes in solution.values())

    def _calculate_relocate_delta(self, node, from_vid, to_vid, solution):
        cost_before = self._get_route_cost(tuple(sorted(solution[from_vid]))) + self._get_route_cost(tuple(sorted(solution[to_vid])))
        cost_after = self._get_route_cost(tuple(sorted([n for n in solution[from_vid] if n != node]))) + self._get_route_cost(tuple(sorted(solution[to_vid] + [node])))
        return cost_after - cost_before

    def _calculate_swap_delta(self, c_a, c_b, v_a, v_b, solution):
        cost_before = self._get_route_cost(tuple(sorted(solution[v_a]))) + self._get_route_cost(tuple(sorted(solution[v_b])))
        cost_after = self._get_route_cost(tuple(sorted([n for n in solution[v_a] if n != c_a] + [c_b]))) + self._get_route_cost(tuple(sorted([n for n in solution[v_b] if n != c_b] + [c_a])))
        return cost_after - cost_before

    def _calculate_k_opt_delta(self, customers, from_v, to_v, solution):
        affected_vids = set(from_v) | set(to_v)
        cost_before = sum(self._get_route_cost(tuple(sorted(solution[vid]))) for vid in affected_vids)
        
        temp_sol = self._apply_k_opt_to_temp(solution, customers, from_v, to_v)
        cost_after = sum(self._get_route_cost(tuple(sorted(temp_sol[vid]))) for vid in affected_vids)
        return cost_after - cost_before
    
    def _calculate_supernode_relocate_delta(self, supernode, from_key, to_key, solution):
        cost_before = self._get_route_cost(tuple(sorted(solution[from_key]))) + self._get_route_cost(tuple(sorted(solution[to_key])))
        
        from_nodes_after = tuple(sorted([n for n in solution[from_key] if n not in supernode]))
        to_nodes_after = tuple(sorted(list(solution[to_key]) + supernode))
        
        cost_after = self._get_route_cost(from_nodes_after) + self._get_route_cost(to_nodes_after)
        return cost_after - cost_before

    def _calculate_supernode_swap_delta(self, supernode1, supernode2, key1, key2, solution):
        cost_before = self._get_route_cost(tuple(sorted(solution[key1]))) + self._get_route_cost(tuple(sorted(solution[key2])))
        
        nodes1_after = tuple(sorted([n for n in solution[key1] if n not in supernode1] + supernode2))
        nodes2_after = tuple(sorted([n for n in solution[key2] if n not in supernode2] + supernode1))
        
        cost_after = self._get_route_cost(nodes1_after) + self._get_route_cost(nodes2_after)
        return cost_after - cost_before
    
    def _apply_move_to_copy(self, solution, move):
        new_sol = copy.deepcopy(solution)
        m_type = move.get('type')

        if 'relocate' in m_type:
            cust, from_v, to_v = move['customer'], move['from'], move['to']
            if cust in new_sol[from_v]: new_sol[from_v].remove(cust)
            new_sol[to_v].append(cust)
        elif 'swap' in m_type:
            (c1, c2), (v1, v2) = move['customers'], move['vehicles']
            if c1 in new_sol[v1]: new_sol[v1].remove(c1)
            if c2 in new_sol[v2]: new_sol[v2].remove(c2)
            new_sol[v1].append(c2)
            new_sol[v2].append(c1)
        elif 'opt' in m_type:
            customers, from_v, to_v = move['customers'], move['from_v'], move['to_v']
            for i, cust in enumerate(customers):
                if cust in new_sol[from_v[i]]: new_sol[from_v[i]].remove(cust)
            for i, cust in enumerate(customers):
                new_sol[to_v[i]].append(cust)

        return {k: v for k, v in new_sol.items() if v}

    def _get_node_to_vehicle_map(self, solution):
        return {c: vid for vid, cluster in solution.items() for c in cluster}

    def _log_gap_event(self, elapsed_time, current_cost, event_type):
        gap = ((current_cost - self.optimal_estimated_cost) / self.optimal_estimated_cost * 100) if self.optimal_estimated_cost else 0
        self.estimator_gap_history.append((elapsed_time, gap, event_type))
        print(f" > New Local Best Found @ {elapsed_time:.2f}s | Delta {((current_cost - self.optimal_estimated_cost) if self.optimal_estimated_cost else 0):,.2f} via {event_type}")

    def _update_and_log_new_best(self, new_best_solution, new_best_cost, elapsed_time):
        self._log_gap_event(elapsed_time, new_best_cost, 'new_best')
        self.global_bests_for_lkh.append((elapsed_time, new_best_solution))
        gap_percent = self.estimator_gap_history[-1][1] if self.estimator_gap_history else 0.0
        print(f" > New Global Best Found @ {elapsed_time:.2f}s | Est. Cost: {new_best_cost:,.2f} (Gap: {gap_percent:+.2f}%)")

    def _calculate_final_true_costs(self):
        print(" [Phase] Calculating final true costs for all best solutions found...")
        if not self.global_bests_for_lkh:
            initial_sol = generate_feasible_solution_regret(self.customers, self.depot_coord, self.num_vehicles, self.capacity)
            if initial_sol: self.global_bests_for_lkh.append((0.0, initial_sol))
            else: return

        for t, sol in self.global_bests_for_lkh:
            true_cost = get_true_VRP_cost(sol, self.all_customer_data_orig, self.depot_id)
            if self.optimal_cost > 0:
                self.true_cost_gap_history.append((t, ((true_cost - self.optimal_cost) / self.optimal_cost * 100)))
        
        self.best_true_solution = self.global_bests_for_lkh[-1][1] if self.global_bests_for_lkh else {}
        self.best_true_cost = get_true_VRP_cost(self.best_true_solution, self.all_customer_data_orig, self.depot_id)

    def _generate_reports(self, overall_start_time: float):
        final_est_cost = self._calculate_total_estimated_cost(self.best_true_solution)
        final_gap = ((final_est_cost - self.optimal_estimated_cost) / self.optimal_estimated_cost * 100) if self.optimal_estimated_cost != 0 else 0
        self.timing_results['4. Total Time'] = time.time() - overall_start_time
        algo_name = "ThreeCampaign-Tabu-v1"
        output_path_txt = os.path.join(self.output_dir, f"{self.basename}_solution.txt")
        write_solution_file(self.basename, self.best_true_solution, final_est_cost, self.best_true_cost, self.optimal_routes, self.optimal_cost, output_path_txt, self.timing_results, algo_name, opt_est_cost=self.optimal_estimated_cost, final_gap=final_gap)
        output_path_map = os.path.join(self.output_dir, f"{self.basename}_map.png")
        plot_solution(self.basename, self.depot_coord, self.best_true_solution, self.customers, output_path_map, self.optimal_cost, final_est_cost, self.best_true_cost, algo_name, optimal_routes=self.optimal_routes)
        output_path_gap_trend = os.path.join(self.output_dir, f"{self.basename}_gap_trend.png")
        self._plot_gap_trend_with_details(self.basename, self.estimator_gap_history, self.true_cost_gap_history, output_path_gap_trend, algo_name)
        verify_solution_feasibility(self.best_true_solution, self.customers, self.capacity, self.basename)

    def _plot_gap_trend_with_details(self, basename, est_gap_history, true_cost_gap_history, output_path, algo_name):
        """
        Convergence profile with clean, compact labels and complete coverage of move types.
    
        - Shows all improving moves logged via self._log_gap_event(..., event_type)
        - Groups markers by campaign and operator with concise labels
        - Includes true-cost gap overlay if available
        - Keeps legend compact and placed outside the plot area
        """
        if not est_gap_history:
            return
    
        # Build DataFrame from existing reporting tuples: (time, gap, type)
        df = pd.DataFrame(est_gap_history, columns=['time', 'gap', 'type'])
    
        # Label mapping: collapse verbose/variant event types into cleaner, compact groups
        def map_label(evt: str) -> str:
            # Global markers
            if evt == 'new_best':
                return 'Best'
            if evt == 'Perturbation':
                return 'Perturb'
            # Campaign 1
            if evt.startswith('C1_'):
                # Supernode sub-ops
                if 'SN' in evt or 'supernode' in evt:
                    return 'C1‑Supernode'
                # Probabilistic relocate polish
                if '1-opt_polish' in evt or 'polish' in evt:
                    return 'C1‑Polish'
                # Targeted swap
                if 'targeted_swap' in evt or '_swap' in evt:
                    return 'C1‑Swap'
                # k-opt (including 3-opt local)
                if '-opt' in evt:
                    return 'C1‑k‑opt'
                return 'C1‑Other'
            # Campaign 2 (Polar shape-tuning)
            if evt.startswith('C2_'):
                if 'polar_angular' in evt or 'angular' in evt:
                    return 'C2‑Angular'
                if 'polar_radial' in evt or 'radial' in evt:
                    return 'C2‑Radial'
                return 'C2‑Other'
            # Campaign 3 (Border refinement)
            if evt.startswith('C3_'):
                if 'border_swap' in evt or '_swap' in evt:
                    return 'C3‑Swap'
                if 'border_reloc' in evt or 'reloc' in evt:
                    return 'C3‑Reloc'
                return 'C3‑Other'
            # Campaign 4 (Global search)
            if evt.startswith('C4_'):
                if '3-opt' in evt or '-opt' in evt:
                    return 'C4‑3‑opt'
                return 'C4‑Other'
            return 'Other'
    
        df['label'] = df['type'].apply(map_label)
    
        # Style palette by label (compact, readable, consistent)
        style_by_label = {
            # Global
            'Best':        {'color': '#DAA520', 'marker': '*', 's': 140, 'zorder': 10, 'edgecolors': 'black'},
            'Perturb':     {'color': '#E31A1C', 'marker': 'X', 's': 120, 'zorder': 9},
    
            # C1 (orange family)
            'C1‑Supernode': {'color': '#FDBF6F', 'marker': 's', 's': 55, 'zorder': 7},
            'C1‑Polish':    {'color': '#FF7F00', 'marker': 'o', 's': 45, 'zorder': 7},
            'C1‑Swap':      {'color': '#FB9A99', 'marker': 'D', 's': 50, 'zorder': 7},
            'C1‑k‑opt':     {'color': '#FF7F00', 'marker': 'P', 's': 65, 'zorder': 7},
    
            # C2 (green family)
            'C2‑Angular':   {'color': '#33A02C', 'marker': 'o', 's': 55, 'zorder': 6},
            'C2‑Radial':    {'color': '#B2DF8A', 'marker': '^', 's': 55, 'zorder': 6},
    
            # C3 (blue family)
            'C3‑Swap':      {'color': '#1F78B4', 'marker': 'D', 's': 55, 'zorder': 6},
            'C3‑Reloc':     {'color': '#66C2A5', 'marker': 'v', 's': 55, 'zorder': 6},
    
            # C4 (purple family)
            'C4‑3‑opt':     {'color': '#6A3D9A', 'marker': '^', 's': 80, 'zorder': 8, 'edgecolors': 'black'},
    
            # Fallback
            'C1‑Other':     {'color': '#FF7F00', 'marker': '.', 's': 35, 'zorder': 5},
            'C2‑Other':     {'color': '#33A02C', 'marker': '.', 's': 35, 'zorder': 5},
            'C3‑Other':     {'color': '#1F78B4', 'marker': '.', 's': 35, 'zorder': 5},
            'C4‑Other':     {'color': '#6A3D9A', 'marker': '.', 's': 35, 'zorder': 5},
            'Other':        {'color': '#888888', 'marker': '.', 's': 30, 'zorder': 4},
        }
    
        # Preferred legend order (compact grouping by campaign)
        preferred_order = [
            'Best', 'Perturb',
            'C1‑Supernode', 'C1‑Polish', 'C1‑Swap', 'C1‑k‑opt',
            'C2‑Angular', 'C2‑Radial',
            'C3‑Swap', 'C3‑Reloc',
            'C4‑3‑opt',
            'C1‑Other', 'C2‑Other', 'C3‑Other', 'C4‑Other', 'Other'
        ]
    
        # Figure
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
    
        # Base line: full timeline of estimator gap
        ax.plot(df['time'], df['gap'], color='silver', linestyle='-', linewidth=0.9, zorder=3, label='_hidden_base')
    
        # Scatter by label (aggregated), so legend stays clean and non-duplicated
        labels_present = [lbl for lbl in preferred_order if lbl in set(df['label'])]
        for lbl in labels_present:
            subset = df[df['label'] == lbl]
            style = style_by_label.get(lbl, style_by_label['Other'])
            # Avoid kwargs that matplotlib.scatter doesn't accept if missing
            kwargs = {k: v for k, v in style.items()}
            ax.scatter(subset['time'], subset['gap'], label=lbl, **kwargs)
    
        # True-cost overlay (if computed)
        if true_cost_gap_history:
            df_true = pd.DataFrame(true_cost_gap_history, columns=['time', 'gap'])
            ax.plot(df_true['time'], df_true['gap'],
                    color='darkviolet', linestyle=':', marker='x', markersize=6,
                    label='True gap', zorder=5)
    
        # Optimal estimated cost reference
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, label='Opt est', zorder=2)
    
        # Titles and axes
        ax.set_title(f'{algo_name} Convergence Profile for {basename}', fontsize=15)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Percentage gap (%)', fontsize=12)
    
        # Grid and layout
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        # Legend outside the plot, compact and readable
        handles, labels = ax.get_legend_handles_labels()
        # Deduplicate legend entries while preserving order
        seen = set()
        filtered = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l)) and not l.startswith('_hidden')]
        if filtered:
            ax.legend([h for h, _ in filtered], [l for _, l in filtered],
                      loc='center left', bbox_to_anchor=(1.0, 0.5),
                      fontsize='small', frameon=False)
    
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()