"""
Final VRP Estimator 3-Way Comparative Analysis

This script performs a rapid, large-scale comparison between three estimators:
1. The classic Vinel & Silva formula.
2. A trained single-output (Alpha-Only) ML model.
3. A trained multi-output (Alpha-Beta) ML model.

It processes all instances in parallel, times each estimator, and generates
a full suite of comparative plots and a summary Excel file.
"""
import os
import re
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import concurrent.futures
from functools import partial

# --- Ensure vrp_utils can be imported ---
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

try:
    from vrp_utils import parse_vrp, parse_sol, calculate_estimated_cost, _calculate_held_karp, calculate_features_and_mst_length, get_true_VRP_cost
except ImportError:
    print("FATAL: Could not import from vrp_utils.py.")
    sys.exit(1)

# --- Configuration ---
DATA_DIR = SCRIPT_DIR.parent
INSTANCE_DIR = DATA_DIR / "instances"
SOLUTION_DIR = DATA_DIR / "solutions"
ML_MODEL_DIR = SCRIPT_DIR / "ML_model"
ALPHA_MODEL_PATH = ML_MODEL_DIR / "alpha_predictor_model.joblib"
ALPHA_BETA_MODEL_PATH = ML_MODEL_DIR / "alpha_beta_predictor_model.joblib"
OUTPUT_DIR = SCRIPT_DIR / "Final_Comparison_3-Way"

# --- Mappings for Plotting ---
CAT_A_MAP = {1: 'Random', 2: 'Centered', 3: 'Cornered'}
CAT_B_MAP = {1: 'Random', 2: 'Clustered', 3: 'Random-Clustered'}
CAT_C_MAP = {1: 'Unitary', 2: 'Sm/LgCV', 3: 'Sm/SmCV', 4: 'Lg/LgCV', 5: 'Lg/SmCV', 6: 'Quadrant', 7: 'Mixed'}
CAT_D_MAP = {1: 'V.Short', 2: 'Short', 3: 'Medium', 4: 'Long', 5: 'V.Long', 6: 'Ultra Long'}


def analyze_instance(paths, models):
    """Performs the 3-way comparative analysis for a single VRP instance."""
    instance_path, sol_path = paths
    try:
        _, depot_id, coords, demands = parse_vrp(instance_path)
        optimal_routes, benchmark_cost = parse_sol(sol_path)
        
        all_customer_data = {nid: {'coords': c, 'demand': demands.get(nid, 0)} for nid, c in coords.items()}
        depot_coord = coords[depot_id]
        optimal_routes_partition = {i: route for i, route in enumerate(optimal_routes)}
        
        # --- Time and Calculate the "True" LKH-3 Benchmark Cost ---
        start_time_lkh3 = time.perf_counter()
        lkh3_true_cost = get_true_VRP_cost(optimal_routes_partition, all_customer_data, depot_id)
        time_lkh3 = time.perf_counter() - start_time_lkh3

        # --- Estimator 1: Vinel Calculation ---
        start_time_vinel = time.perf_counter()
        vinel_cost = calculate_estimated_cost(optimal_routes_partition, all_customer_data, depot_coord, mode='composite')
        time_vinel = time.perf_counter() - start_time_vinel

        # --- Pre-calculate features for both ML models ---
        precalc_start_time = time.perf_counter()
        route_features = {}
        for i, route in optimal_routes_partition.items():
            if not route: continue
            route_coords_list = [depot_coord] + [all_customer_data[nid]['coords'] for nid in route]
            n = len(route_coords_list)
            if n < 10:
                route_features[i] = {'type': 'small', 'cost': _calculate_held_karp(route_coords_list)}
            else:
                features, mst = calculate_features_and_mst_length(route_coords_list)
                if features and mst > 0:
                    route_features[i] = {'type': 'large', 'features': features, 'mst': mst}
        precalc_time = time.perf_counter() - precalc_start_time

        # --- Estimator 2: Alpha-Only ML Model Calculation ---
        start_time_alpha = time.perf_counter()
        alpha_only_cost = 0
        alpha_model = models['alpha_only']
        feature_cols_alpha = alpha_model.feature_name_
        for i, route in optimal_routes_partition.items():
            if i not in route_features: continue
            if route_features[i]['type'] == 'small':
                alpha_only_cost += route_features[i]['cost']
            else:
                feature_df = pd.DataFrame([route_features[i]['features']])[feature_cols_alpha]
                predicted_alpha = alpha_model.predict(feature_df)[0]
                alpha_only_cost += predicted_alpha * route_features[i]['mst']
        time_alpha = time.perf_counter() - start_time_alpha + precalc_time

        # --- Estimator 3: Alpha-Beta ML Model Calculation ---
        start_time_alpha_beta = time.perf_counter()
        alpha_beta_cost = 0
        alpha_beta_model = models['alpha_beta']
        feature_cols_ab = alpha_beta_model.estimators_[0].feature_name_
        for i, route in optimal_routes_partition.items():
            if i not in route_features: continue
            if route_features[i]['type'] == 'small':
                alpha_beta_cost += route_features[i]['cost']
            else:
                feature_df = pd.DataFrame([route_features[i]['features']])[feature_cols_ab]
                prediction = alpha_beta_model.predict(feature_df)
                predicted_alpha, predicted_beta = prediction[0]
                alpha_beta_cost += (predicted_alpha * route_features[i]['mst']) + predicted_beta
        time_alpha_beta = time.perf_counter() - start_time_alpha_beta + precalc_time
        
        # --- Calculate Errors ---
        vinel_error = ((vinel_cost - benchmark_cost) / benchmark_cost * 100) if benchmark_cost > 0 else 0
        alpha_only_error = ((alpha_only_cost - benchmark_cost) / benchmark_cost * 100) if benchmark_cost > 0 else 0
        alpha_beta_error = ((alpha_beta_cost - benchmark_cost) / benchmark_cost * 100) if benchmark_cost > 0 else 0
        
        match = re.search(r'XML100_(\d)(\d)(\d)(\d)', instance_path.name)
        cat_a, cat_b, cat_c, cat_d = [int(g) for g in match.groups()] if match else [0,0,0,0]

        return {
            'Instance': instance_path.stem, 
            'Benchmark Cost': benchmark_cost,
            'LKH-3 Recalculated Cost': lkh3_true_cost,
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
            'Route Size (D)': CAT_D_MAP.get(cat_d, 'Unknown'),
        }
    except Exception as e:
        return f"ERROR processing {instance_path.name}: {e}"


def generate_error_plots(df, category_col, category_name, category_map):
    """Generates box plots comparing the error of all three estimators."""
    df_melted = df.melt(
        id_vars=[category_col], value_vars=['Vinel Error (%)', 'Alpha-Only Error (%)', 'Alpha-Beta Error (%)'],
        var_name='Estimator Type', value_name='Error (%)'
    )
    df_melted['Estimator Type'] = df_melted['Estimator Type'].str.replace(' Error (%)', '')
    plt.figure(figsize=(14, 8))
    order = [category_map[k] for k in sorted(category_map.keys())] if category_map else None
    sns.boxplot(data=df_melted, x=category_col, y='Error (%)', hue='Estimator Type',
                palette={'Vinel': '#8da0cb', 'Alpha-Only': '#fc8d62', 'Alpha-Beta': '#66c2a5'}, order=order)
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
    """Generates box plots comparing the speed of all three estimators and LKH-3."""
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


def main():
    """Main function to orchestrate the 3-way comparative analysis."""
    print("Starting 3-Way Estimator Comparative Analysis...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Results will be saved in: {OUTPUT_DIR}")

    try:
        alpha_model = joblib.load(ALPHA_MODEL_PATH)
        alpha_beta_model = joblib.load(ALPHA_BETA_MODEL_PATH)
        models = {'alpha_only': alpha_model, 'alpha_beta': alpha_beta_model}
        print("✅ Successfully loaded both ML models.")
    except FileNotFoundError as e:
        print(f"FATAL: Could not load a required model file. Error: {e}")
        return

    all_instance_paths = sorted([f for f in INSTANCE_DIR.glob('XML100_*.vrp')])
    tasks = [(p, SOLUTION_DIR / f"{p.stem}.sol") for p in all_instance_paths if (SOLUTION_DIR / f"{p.stem}.sol").exists()]
    print(f"Found {len(tasks)} instance/solution pairs to analyze.")

    results_list = []
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Starting analysis with {num_workers} parallel workers...")
    
    analysis_func = partial(analyze_instance, models=models)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(analysis_func, tasks), total=len(tasks), desc="Analyzing Instances"):
            if isinstance(result, dict):
                results_list.append(result)
            elif isinstance(result, str):
                print(f"\n{result}")

    if not results_list:
        print("No results were generated. Exiting."); return

    df_results = pd.DataFrame(results_list)
    df_results.to_excel(OUTPUT_DIR / "final_3-way_comparison_results.xlsx", index=False)
    print(f"\nFull results saved to: {OUTPUT_DIR / 'final_3-way_comparison_results.xlsx'}")

    print("\n" + "="*80)
    print("TIMING ANALYSIS (Average per Instance)")
    print("="*80)
    print(f"Vinel Estimator    : {df_results['Time (Vinel)'].mean() * 1000:.2f} ms")
    print(f"Alpha-Only Model   : {df_results['Time (Alpha-Only)'].mean() * 1000:.2f} ms")
    print(f"Alpha-Beta Model   : {df_results['Time (Alpha-Beta)'].mean() * 1000:.2f} ms")
    print(f"LKH-3 Solver       : {df_results['Time (LKH-3)'].mean() * 1000:.2f} ms")

    print("\n" + "="*80)
    print("ACCURACY ANALYSIS (Mean Absolute Error vs. Benchmark)")
    print("="*80)
    print(f"Vinel Estimator    : {df_results['Vinel Error (%)'].abs().mean():.2f}%")
    print(f"Alpha-Only Model   : {df_results['Alpha-Only Error (%)'].abs().mean():.2f}%")
    print(f"Alpha-Beta Model   : {df_results['Alpha-Beta Error (%)'].abs().mean():.2f}%")

    # Add a check for LKH-3's cost match
    if 'LKH-3 Recalculated Cost' in df_results.columns:
        df_results['LKH-3 Error (%)'] = ((df_results['LKH-3 Recalculated Cost'] - df_results['Benchmark Cost']) / df_results['Benchmark Cost']) * 100
        print("\n" + "="*80)
        print("LKH-3 SOLUTION VERIFICATION")
        print("="*80)
        print(f"LKH-3 Avg. Error vs. .sol File: {df_results['LKH-3 Error (%)'].abs().mean():.2f}%")
        print(f"Number of instances with a perfect match: {sum(df_results['LKH-3 Error (%)'].abs() < 1e-6)}/{len(df_results)}")

    print("\nGenerating plots...")
    df_results['Aggregate'] = 'All Instances'
    generate_error_plots(df_results, 'Aggregate', 'Aggregate Analysis', None)
    generate_error_plots(df_results, 'Depot (A)', 'Depot Type', CAT_A_MAP)
    generate_error_plots(df_results, 'Customers (B)', 'Customer Distribution', CAT_B_MAP)
    generate_error_plots(df_results, 'Demand (C)', 'Demand Profile', CAT_C_MAP)
    generate_error_plots(df_results, 'Route Size (D)', 'Avg. Route Size', CAT_D_MAP)
    generate_timing_plot(df_results)
    
    print("\nAnalysis complete. All plots and reports have been saved.")

if __name__ == "__main__":
    main()