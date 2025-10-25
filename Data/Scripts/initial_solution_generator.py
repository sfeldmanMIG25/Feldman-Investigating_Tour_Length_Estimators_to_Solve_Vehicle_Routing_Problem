# initial_solution_generator.py
"""
VRP Initial Solution Generator (Regret Heuristic with First-Fit Fallback)

This script provides a robust method for creating a feasible initial solution.
It first attempts a high-quality Regret-2 heuristic and, if that fails on a
tightly constrained instance, it falls back to a simpler but more robust
First-Fit heuristic.
"""
import numpy as np
import random

def get_insertion_cost(customer_coords, route, customer_data, depot_coord):
    """Calculates the cost of inserting a customer into a route."""
    if not route:
        return 2 * np.linalg.norm(np.array(customer_coords) - np.array(depot_coord))
    route_coords = np.array([customer_data[rcid]['coords'] for rcid in route])
    return np.min(np.linalg.norm(route_coords - np.array(customer_coords), axis=1))

def _generate_regret(customer_data, num_vehicles, capacity):
    """The core Regret-2 insertion heuristic logic."""
    solution = {i + 1: [] for i in range(num_vehicles)}
    vehicle_loads = {i + 1: 0 for i in range(num_vehicles)}
    unassigned_customers = set(customer_data.keys())
    depot_coord = (0, 0) # Placeholder, depot coord not needed for regret calc logic

    while unassigned_customers:
        insertion_options = []
        for cid in unassigned_customers:
            node_demand = customer_data[cid]['demand']
            node_coords = customer_data[cid]['coords']
            costs = []
            for vid, route in solution.items():
                if vehicle_loads[vid] + node_demand <= capacity:
                    cost = get_insertion_cost(node_coords, route, customer_data, depot_coord)
                    costs.append({'vid': vid, 'cost': cost})
            if costs:
                costs.sort(key=lambda x: x['cost'])
                insertion_options.append({'cid': cid, 'costs': costs})

        if not insertion_options:
            raise RuntimeError("Regret heuristic failed: No feasible insertions found.")

        for option in insertion_options:
            regret = option['costs'][1]['cost'] - option['costs'][0]['cost'] if len(option['costs']) > 1 else float('inf')
            option['regret'] = regret
            
        insertion_options.sort(key=lambda x: x['regret'], reverse=True)
        best_option = insertion_options[0]
        
        customer_to_insert = best_option['cid']
        target_vehicle_id = best_option['costs'][0]['vid']
        
        solution[target_vehicle_id].append(customer_to_insert)
        vehicle_loads[target_vehicle_id] += customer_data[customer_to_insert]['demand']
        unassigned_customers.remove(customer_to_insert)

    return {vid: route for vid, route in solution.items() if route}

def _generate_first_fit(customer_data, num_vehicles, capacity):
    """A simple First-Fit fallback heuristic."""
    print("WARNING: Regret heuristic failed. Falling back to robust First-Fit method.")
    solution = {i + 1: [] for i in range(num_vehicles)}
    vehicle_loads = {i + 1: 0 for i in range(num_vehicles)}
    
    # Shuffle customers to avoid consistently bad solutions for ordered data
    unassigned_customers = list(customer_data.keys())
    random.shuffle(unassigned_customers)
    
    for cid in unassigned_customers:
        demand = customer_data[cid]['demand']
        placed = False
        for vid in solution.keys():
            if vehicle_loads[vid] + demand <= capacity:
                solution[vid].append(cid)
                vehicle_loads[vid] += demand
                placed = True
                break
        if not placed:
            raise RuntimeError("First-Fit fallback failed: A customer could not be placed in any vehicle. The problem may be infeasible with the given number of vehicles.")

    return {vid: route for vid, route in solution.items() if route}

def generate_feasible_solution_regret(customer_data, depot_coord, num_vehicles, capacity):
    """
    Main entry point. Tries Regret-2 heuristic first, then falls back to First-Fit.
    """
    try:
        # Pass a copy of customer_data to avoid modification issues
        return _generate_regret(dict(customer_data), num_vehicles, capacity)
    except RuntimeError as e:
        print(f"{e}")
        # If Regret fails, try the simpler, more robust method
        return _generate_first_fit(dict(customer_data), num_vehicles, capacity)