import os
import sys
import time
import math
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt

# --- Ensure required files can be imported ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

try:
    from vrp_utils import (
        parse_vrp,
        parse_sol,
    )
except ImportError as e:
    print(f"FATAL: Could not import a required module. Error: {e}")
    sys.exit(1)

# --- Configuration ---
OUTPUT_BASE_DIR = DATA_DIR
SOLUTION_DIRS_MEDIUM = [d for d in OUTPUT_BASE_DIR.iterdir() if d.name.startswith('solutions_medium_') and d.is_dir()]
SOLUTION_DIRS_LARGE = [d for d in OUTPUT_BASE_DIR.iterdir() if d.name.startswith('solutions_large_') and d.is_dir()]
ALL_SOLUTION_DIRS = SOLUTION_DIRS_MEDIUM + SOLUTION_DIRS_LARGE

# --- Visualization Function ---

def plot_solution_from_files(sol_path):
    """
    Parses a .sol and .vrp file, generates a plot of the solution,
    and saves the plot as a PNG in the same directory.
    """
    try:
        sol_basename = sol_path.stem
        
        # Determine the corresponding instance directory
        if 'solutions_medium' in sol_path.parent.name:
            instance_base_dir = DATA_DIR / 'instances_medium'
        elif 'solutions_large' in sol_path.parent.name:
            instance_base_dir = DATA_DIR / 'instances_large'
        else:
            return f"Skipped {sol_path.name}: Parent directory name not recognized."
            
        instance_path = instance_base_dir / f"{sol_basename}.vrp"
        
        if not instance_path.exists():
            return f"Skipped {sol_path.name}: Corresponding VRP instance not found."
            
        # Parse the instance and solution files
        capacity, depot_id, coords, demands = parse_vrp(instance_path)
        sol_routes, sol_cost = parse_sol(sol_path, False)
        
        # Get heuristic name from directory name
        heuristic_name = sol_path.parent.name.split('_')[-1]
        
        plt.figure(figsize=(12, 10))
        cmap = plt.get_cmap('tab20')

        # Plot the depot
        depot_coord = coords[depot_id]
        plt.scatter(depot_coord[0], depot_coord[1], c='red', marker='*', s=250, zorder=10, label='Depot')

        # Plot all customers, color-coded by route
        all_customers = {k: v for k, v in coords.items() if k != depot_id}
        
        customer_coords_list = []
        for route in sol_routes:
            for customer_id in route:
                customer_coords_list.append(all_customers[customer_id])
                
        # This is not a great way to do it. The colors wont match the routes.
        # Let's fix that.
        
        # New plotting logic:
        for i, route in enumerate(sol_routes):
            color = cmap(i % 20)
            if not route:
                continue
            
            route_coords = [coords[cid] for cid in route]
            route_coords_np = np.array(route_coords)
            
            # Plot the customers in this route
            plt.scatter(route_coords_np[:, 0], route_coords_np[:, 1], c=[color], s=30, alpha=0.8, label=f'Route {i+1}')
        
        title = f"Solution for {sol_basename} ({heuristic_name})\nTotal Cost: {sol_cost:,.2f}"
        plt.title(title, fontsize=14)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        
        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), markerscale=2, loc='best')
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        output_path = sol_path.parent / f"{sol_basename}_plot.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return f"Successfully generated plot for {sol_basename}."
        
    except Exception as e:
        plt.close('all') # Ensure any open plot windows are closed on error
        return f"Error processing {sol_path.name}: {e}"

# --- Main Execution ---

def main():
    """Main function to orchestrate the parallel visualization process."""
    
    all_sol_paths = []
    for sol_dir in ALL_SOLUTION_DIRS:
        all_sol_paths.extend(list(sol_dir.glob('*.sol')))
        
    if not all_sol_paths:
        print("FATAL: No solution files found in the specified directories.")
        sys.exit(1)
        
    print(f"Found {len(all_sol_paths)} solution files to visualize.")
    
    num_workers = os.cpu_count() or 1
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(plot_solution_from_files, sol_path): sol_path for sol_path in all_sol_paths}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_sol_paths), desc="Generating Visualizations"):
            result = future.result()
            if result and "Error" in result:
                print(result)
                
    print("\nVisualization process complete.")

if __name__ == '__main__':
    main()