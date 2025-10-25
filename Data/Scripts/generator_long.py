# run_generator_suite.py
"""
Calls the redesigned generator_v2.py to create a representative set of
large VRP instances with targeted average route lengths.
"""
import subprocess
import os
import sys
import concurrent.futures
from tqdm import tqdm

# Get the directory of this script to find generator_v2.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The parent directory of SCRIPT_DIR is the 'data' folder
DATA_DIR = os.path.join(SCRIPT_DIR, '..')

def _generate_single_instance(params, output_dir):
    """
    Helper function to generate a single instance.
    This function is designed to be run in parallel using threads.
    """
    n, d_pos, c_pos, demand_type, capacity_factor, inst_id = params
    instance_name = f"XML{n}_{d_pos}{c_pos}{demand_type}_{capacity_factor}_{inst_id:02d}"

    try:
        # Call the generator_v2.py script
        subprocess.run(
            [
                "python",
                "generator_v2.py",
                str(n),
                str(d_pos),
                str(c_pos),
                str(demand_type),
                str(capacity_factor),
                str(inst_id),
                str(42) # Fixed random seed for reproducibility
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=SCRIPT_DIR # Run the command from the script's directory
        )
        # Move the generated file from the script directory to the correct subfolder
        generated_file = os.path.join(SCRIPT_DIR, f"{instance_name}.vrp")
        os.rename(generated_file, os.path.join(output_dir, f"{instance_name}.vrp"))
        return f"Successfully generated {instance_name}"
    except subprocess.CalledProcessError as e:
        return f"Error generating {instance_name}: {e.stderr.decode()}"
    except FileNotFoundError:
        return f"FATAL: generator_v2.py not found in {SCRIPT_DIR}."

def generate_parallel_instances(
    n_customers_list, 
    depot_pos_options, 
    cust_pos_options, 
    demand_type, 
    capacity_factors, 
    folder_name
):
    """
    Generates instances in parallel using threads and calls generator_v2.py.
    """
    output_dir = os.path.join(DATA_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a list of all parameter tuples
    all_params = []
    for n in n_customers_list:
        for capacity_factor in capacity_factors:
            for d_pos in depot_pos_options:
                for c_pos in cust_pos_options:
                    for inst_id in range(1, 10):
                        # Ensure the minimum route size is always greater than 100
                        # AvgRouteLength = n * capacity_factor
                        if n * capacity_factor < 100:
                            print(f"Skipping combination n={n}, cap_factor={capacity_factor} as avg route length < 100.")
                            continue
                        all_params.append((n, d_pos, c_pos, demand_type, capacity_factor, inst_id))

    num_workers = os.cpu_count() or 1
    print(f"Preparing to generate {len(all_params)} instances using {num_workers} parallel workers...")

    # Use ThreadPoolExecutor for I/O-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_generate_single_instance, params, output_dir) for params in all_params}
        results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_params), desc=f"Generating {folder_name} instances")]

    for result in results:
        if "Error" in result or "FATAL" in result:
            print(result)

    print(f"\nSuccessfully generated {len(all_params)} instances in '{output_dir}'.")
    return len(all_params)

if __name__ == '__main__':
    depot_pos_options = [1, 2, 3] # Random, Centered, Cornered
    cust_pos_options = [1, 2, 3]  # Random, Clustered, Random-Clustered
    demand_type_val = 1           # Unitary Demand

    # Medium instances: Routes with average length of 100-1000 nodes
    # n=1000, min cap_factor is 0.1
    medium_n_customers = [1000]
    medium_capacity_factors = [0.15, 0.25, 0.35, 0.5] 

    print("--- Generating medium instances (avg route length 100-1000) ---")
    generate_parallel_instances(
        medium_n_customers,
        depot_pos_options,
        cust_pos_options,
        demand_type_val,
        medium_capacity_factors,
        "instances_medium"
    )
    
    print("\n" + "="*50 + "\n")

    # Large instances: Routes with average length of 1000+ nodes
    # n=10000, min cap_factor is 0.1
    large_n_customers = [10000]
    large_capacity_factors = [0.15, 0.25, 0.35, 0.5]

    print("--- Generating large instances (avg route length 1000+) ---")
    generate_parallel_instances(
        large_n_customers,
        depot_pos_options,
        cust_pos_options,
        demand_type_val,
        large_capacity_factors,
        "instances_large"
    )