# cost_estimator_large_test_v2.py
"""
Analyzes the performance of the ML estimator by evaluating the cost of
pre-generated solutions. This script reads .sol files and compares the
estimator's cost to the true cost provided in the solution file.
"""
import os
import sys
import time
import joblib
import pandas as pd
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functools import partial

# --- Ensure required files can be imported ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

try:
    from vrp_utils import (
        parse_vrp,
        parse_sol,
        calculate_estimated_cost,
        calculate_features_and_mst_length,
        _calculate_held_karp,
        get_true_VRP_cost
    )
except ImportError as e:
    print(f"FATAL: Could not import a required module. Error: {e}")
    sys.exit(1)

# --- Configuration ---
ML_MODEL_DIR = SCRIPT_DIR / "ML_model"
ALPHA_MODEL_PATH = ML_MODEL_DIR / "alpha_predictor_model.joblib"
ALPHA_BETA_MODEL_PATH = ML_MODEL_DIR / "alpha_beta_predictor_model.joblib"
OUTPUT_DIR = SCRIPT_DIR / "solution_analysis_large_cavdar"

# --- Mappings for Plotting ---
CAT_A_MAP = {1: 'Random', 2: 'Centered', 3: 'Cornered'}
CAT_B_MAP = {1: 'Random', 2: 'Clustered', 3: 'Random-Clustered'}
CAT_C_MAP = {1: 'Unitary', 2: 'Sm/LgCV', 3: 'Sm/SmCV', 4: 'Lg/LgCV', 5: 'Lg/SmCV', 6: 'Quadrant', 7: 'Mixed'}

# New map for the Cavdar convention
CAVDAR_CAP_MAP = {0.15: 'V.Short', 0.25: 'Short', 0.35: 'Medium', 0.5: 'Long'}

# --- Analysis Function for a Single Instance ---
def analyze_single_solution(paths, models):
    """
    Parses a single .sol file, evaluates its cost with multiple estimators,
    and returns a result dictionary. Designed for parallel execution.
    """
    sol_path, instance_base_dir = paths
    alpha_model = models['alpha_only']
    alpha_beta_model = models['alpha_beta']
    
    sol_basename = sol_path.stem
    
    # 1. Parse the VRP instance
    instance_path = instance_base_dir / f"{sol_basename}.vrp"
    if not instance_path.exists():
        return None # Skip if the corresponding instance file doesn't exist

    capacity, depot_id, coords, demands = parse_vrp(instance_path)
    depot_coord = coords[depot_id]
    
    # Create the customer_data dictionary from all parsed coordinates
    # The `all_customer_data` must contain the depot and its coordinates, as LKH needs it.
    all_customer_data = {
        nid: {'coords': c, 'demand': demands.get(nid, 0)}
        for nid, c in coords.items()
    }

    # Determine if the ID offset should be applied
    apply_offset = not ("XML10000" in sol_basename or "XML1000" in sol_basename or "XML100" in sol_basename)

    # 2. Parse the .sol file for the solution and true cost
    sol_routes, sol_cost_from_file = parse_sol(sol_path, apply_offset=apply_offset)
    
    # 3. Build the solution partition for the estimator
    solution_partition = {i: route for i, route in enumerate(sol_routes)}
    
    # --- Time and Calculate the "True" LKH-3 Benchmark Cost ---
    start_time_lkh3 = time.perf_counter()
    lkh3_true_cost = get_true_VRP_cost(solution_partition, all_customer_data, depot_id)
    time_lkh3 = time.perf_counter() - start_time_lkh3
    
    # --- Prepare customer data for estimators (excluding depot) ---
    customer_data_for_estimators = {
        nid: {'coords': c, 'demand': demands.get(nid, 0)}
        for nid, c in coords.items() if nid != depot_id
    }
    
    # --- Estimator 1: Vinel Calculation ---
    start_time_vinel = time.perf_counter()
    vinel_cost = calculate_estimated_cost(solution_partition, customer_data_for_estimators, depot_coord, mode='composite')
    time_vinel = time.perf_counter() - start_time_vinel

    # --- Pre-calculate features for both ML models ---
    start_time_precompute =time.perf_counter()
    route_features = {}
    for i, route in solution_partition.items():
        if not route: continue
        route_coords_list = [depot_coord] + [customer_data_for_estimators[nid]['coords'] for nid in route]
        n = len(route_coords_list)
        if n < 10:
            route_features[i] = {'type': 'small', 'cost': _calculate_held_karp(route_coords_list)}
        else:
            features, mst = calculate_features_and_mst_length(route_coords_list)
            if features and mst > 0:
                route_features[i] = {'type': 'large', 'features': features, 'mst': mst}
    time_precompute = time.perf_counter() - start_time_precompute
    
    # --- Estimator 2: Alpha-Only ML Model Calculation ---
    start_time_alpha = time.perf_counter()
    alpha_only_cost = 0
    feature_cols_alpha = alpha_model.feature_name_
    for i, route in solution_partition.items():
        if i not in route_features: continue
        if route_features[i]['type'] == 'small':
            alpha_only_cost += route_features[i]['cost']
        else:
            feature_df = pd.DataFrame([route_features[i]['features']])[feature_cols_alpha]
            predicted_alpha = alpha_model.predict(feature_df)[0]
            alpha_only_cost += predicted_alpha * route_features[i]['mst']
    time_alpha = time.perf_counter() - start_time_alpha + time_precompute

    # --- Estimator 3: Alpha-Beta ML Model Calculation ---
    start_time_alpha_beta = time.perf_counter()
    alpha_beta_cost = 0
    feature_cols_ab = alpha_beta_model.estimators_[0].feature_name_
    for i, route in solution_partition.items():
        if i not in route_features: continue
        if route_features[i]['type'] == 'small':
            alpha_beta_cost += route_features[i]['cost']
        else:
            feature_df = pd.DataFrame([route_features[i]['features']])[feature_cols_ab]
            prediction = alpha_beta_model.predict(feature_df)
            predicted_alpha, predicted_beta = prediction[0]
            alpha_beta_cost += (predicted_alpha * route_features[i]['mst']) + predicted_beta
    time_alpha_beta = time.perf_counter() - start_time_alpha_beta + time_precompute
    
    # --- Calculate Errors ---
    vinel_error = ((vinel_cost - lkh3_true_cost) / lkh3_true_cost * 100) if lkh3_true_cost > 0 else float('inf')
    alpha_only_error = ((alpha_only_cost - lkh3_true_cost) / lkh3_true_cost * 100) if lkh3_true_cost > 0 else float('inf')
    alpha_beta_error = ((alpha_beta_cost - lkh3_true_cost) / lkh3_true_cost * 100) if lkh3_true_cost > 0 else float('inf')
    
    # Get heuristic and instance type information
    heuristic_name = sol_path.parent.name.split('_')[-1]
    
    # New logic for handling different naming conventions
    if "XML1000" in sol_basename:
        # Handle the Cavdar naming convention
        match = re.search(r'XML10000?_(\d)(\d)(\d)_(\d+\.\d+)', sol_basename)
        if match:
            cat_a, cat_b, cat_c, capacity_val_str = match.groups()
            cat_a, cat_b, cat_c = int(cat_a), int(cat_b), int(cat_c)
            cat_d = CAVDAR_CAP_MAP.get(float(capacity_val_str), 'Unknown')
        else:
            cat_a, cat_b, cat_c, cat_d = 0, 0, 0, 'Unknown'
    else:
        # Handle the original XML100 naming convention
        match = re.search(r'XML100_(\d)(\d)(\d)(\d)', sol_basename)
        if match:
            cat_a, cat_b, cat_c, cat_d_int = [int(g) for g in match.groups()]
            cat_d = CAVDAR_CAP_MAP.get(cat_d_int, 'Unknown') # Re-using the same map for consistency
        else:
            cat_a, cat_b, cat_c, cat_d = 0, 0, 0, 'Unknown'
        
    return {
        'Instance': sol_basename,
        'Heuristic': heuristic_name,
        'True Cost (Benchmark)': lkh3_true_cost,
        'Vinel Est. Cost': vinel_cost,
        'Alpha-Only Est. Cost': alpha_only_cost,
        'Alpha-Beta Est. Cost': alpha_beta_cost,
        'Vinel Error (%)': vinel_error,
        'Alpha-Only Error (%)': alpha_only_error,
        'Alpha-Beta Error (%)': alpha_beta_error,
        'Time (Vinel)': time_vinel,
        'Time (Alpha-Only)': time_alpha,
        'Time (Alpha-Beta)': time_alpha_beta,
        'Time (LKH-3)': time_lkh3,
        'Depot (A)': CAT_A_MAP.get(cat_a, 'Unknown'),
        'Customers (B)': CAT_B_MAP.get(cat_b, 'Unknown'),
        'Demand (C)': CAT_C_MAP.get(cat_c, 'Unknown'),
        'Route Size (D)': cat_d,
    }

# --- Plotting Functions ---
def generate_error_plots(df, category_col, category_name, category_map):
    """Generates box plots comparing the error of all three estimators."""
    # Filter the DataFrame to only include categories defined in the map
    if category_map:
        valid_categories = list(category_map.values())
        df = df[df[category_col].isin(valid_categories)]

    df_melted = df.melt(
        id_vars=[category_col], value_vars=['Vinel Error (%)', 'Alpha-Only Error (%)', 'Alpha-Beta Error (%)'],
        var_name='Estimator Type', value_name='Error (%)'
    )
    df_melted['Estimator Type'] = df_melted['Estimator Type'].str.replace(' Error (%)', '')

    if df_melted.empty:
        print(f"Skipping plot for {category_name} due to no valid data points.")
        return

    plt.figure(figsize=(14, 8))
    
    # Define the palette separately
    estimator_palette = {'Vinel': '#8da0cb', 'Alpha-Only': '#fc8d62', 'Alpha-Beta': '#66c2a5'}
    
    # Use order if a category map is provided
    order = [category_map[k] for k in sorted(category_map.keys())] if category_map else None

    sns.boxplot(data=df_melted, x=category_col, y='Error (%)', hue='Estimator Type',
                palette=estimator_palette, order=order)

    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.title(f'Estimator Error Comparison by {category_name}', fontsize=16)
    plt.ylabel('Error vs. Benchmark Cost (%)', fontsize=12)
    plt.xlabel(category_name, fontsize=12)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"plot_error_by_{category_name.replace(' ', '_').lower()}.png")
    plt.close()

def generate_timing_plot(df):
    """Generates box plots comparing the speed of all estimators and LKH-3."""
    df_melted = df.melt(
        value_vars=['Time (Vinel)', 'Time (Alpha-Only)', 'Time (Alpha-Beta)', 'Time (LKH-3)'],
        var_name='Estimator Type', value_name='Time (s)'
    )
    df_melted['Estimator Type'] = df_melted['Estimator Type'].str.replace('Time (', '').str.replace(')', '')
    df_melted['Time (ms)'] = df_melted['Time (s)'] * 1000
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_melted, x='Estimator Type', y='Time (ms)',
                palette={'Vinel': '#8da0cb', 'Alpha-Only': '#fc8d62', 'Alpha-Beta': '#66c2a5', 'LKH-3': '#a6d854'})
    plt.title('Estimator Performance: Time per Instance (vs. LKH-3)', fontsize=16)
    plt.ylabel('Computation Time per Instance (milliseconds)', fontsize=12)
    plt.xlabel('Estimator Type', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_timing_comparison.png")
    plt.close()

def generate_scatter_plots(df):
    """
    Generates two scatter plots comparing all estimator models against the true cost,
    separated by instance size (medium vs. large), with color-coding by heuristic
    and marker shapes by model type.
    """
    # Create a 'Size' column to categorize instances
    df['Size'] = 'Medium'
    df.loc[df['Instance'].str.contains('XML10000?'), 'Size'] = 'Large'
    
    # Melt the DataFrame to prepare for multi-model plotting
    df_melted = df.melt(
        id_vars=['Instance', 'Heuristic', 'True Cost (Benchmark)', 'Size'],
        value_vars=['Vinel Est. Cost', 'Alpha-Only Est. Cost', 'Alpha-Beta Est. Cost'],
        var_name='Model Type',
        value_name='Estimated Cost'
    )
    
    # Rename model types for clarity in the plot legend
    df_melted['Model Type'] = df_melted['Model Type'].replace({
        'Vinel Est. Cost': 'Composite',
        'Alpha-Only Est. Cost': 'MST-Alpha',
        'Alpha-Beta Est. Cost': 'MST-AlphaBeta'
    })
    
    # Generate plots for each instance size
    for size, sub_df in df_melted.groupby('Size'):
        plt.figure(figsize=(10, 10))
        
        # Plot the scatter points with hue for heuristics and style for model types
        sns.scatterplot(
            data=sub_df,
            x='True Cost (Benchmark)',
            y='Estimated Cost',
            hue='Heuristic',
            style='Model Type',
            alpha=0.6
        )
        
        # Plot the y=x line for perfect estimation
        max_val = max(sub_df['True Cost (Benchmark)'].max(), sub_df['Estimated Cost'].max())
        min_val = min(sub_df['True Cost (Benchmark)'].min(), sub_df['Estimated Cost'].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (Perfect Estimation)')
        
        # Set plot labels and title
        plt.title(f'Estimated vs. True Cost for {size} Instances', fontsize=16)
        plt.xlabel('True Cost (Benchmark)', fontsize=12)
        plt.ylabel('Estimated Cost', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # Save the figure
        plt.savefig(OUTPUT_DIR / f'plot_scatter_{size.lower()}.png')
        plt.close()

# --- Main Execution ---
def main():
    """Main function to orchestrate the parallel analysis."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    try:
        alpha_model = joblib.load(ALPHA_MODEL_PATH)
        alpha_beta_model = joblib.load(ALPHA_BETA_MODEL_PATH)
        models = {'alpha_only': alpha_model, 'alpha_beta': alpha_beta_model}
        print("✅ Successfully loaded the ML estimator models.")
    except FileNotFoundError as e:
        print(f"FATAL: ML model not found. Error: {e}")
        sys.exit(1)

    # Discover all solution directories
    sol_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('solutions_')]
    if not sol_dirs:
        print("FATAL: No solution directories found in the specified path.")
        sys.exit(1)
        
    all_tasks = []
    for sol_dir in sol_dirs:
        sol_files = list(sol_dir.glob('*.sol'))
        if 'medium' in sol_dir.name:
            instance_base_dir = DATA_DIR / 'instances_medium'
        elif 'large' in sol_dir.name:
            instance_base_dir = DATA_DIR / 'instances_large'
        else:
            continue
            
        for sol_file in sol_files:
            all_tasks.append(((sol_file, instance_base_dir), models))

    if not all_tasks:
        print("No solution files found to analyze.")
        sys.exit(0)

    print(f"Found {len(all_tasks)} solution files to analyze.")
    
    results_list = []
    num_workers = os.cpu_count() or 1
    
    analysis_func = partial(analyze_single_solution, models=models)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(analysis_func, [t[0] for t in all_tasks]), total=len(all_tasks), desc="Analyzing Solutions"):
            if result:
                results_list.append(result)
                
    if not results_list:
        print("Analysis completed, but no results were generated.")
        sys.exit(0)

    df = pd.DataFrame(results_list)
    output_path = OUTPUT_DIR / "estimator_performance_summary.xlsx"
    df.to_excel(output_path, index=False)
    
    print(f"\nAnalysis complete. Summary saved to: {output_path}")

    # Generate plots
    print("\nGenerating plots...")
    df['Aggregate'] = 'All Instances'
    generate_error_plots(df, 'Aggregate', 'Aggregate Analysis', None)
    generate_error_plots(df, 'Depot (A)', 'Depot Type', CAT_A_MAP)
    generate_error_plots(df, 'Customers (B)', 'Customer Distribution', CAT_B_MAP)
    generate_error_plots(df, 'Demand (C)', 'Demand Profile', CAT_C_MAP)
    generate_error_plots(df, 'Route Size (D)', 'Avg. Route Size', CAVDAR_CAP_MAP)
    generate_timing_plot(df)
    generate_scatter_plots(df)
    
    print("\nAnalysis complete. All plots and reports have been saved.")

if __name__ == '__main__':
    main()